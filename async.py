from __future__ import print_function
import logging
import numpy as np
import tensorflow as tf
import six.moves.queue as queue
import threading
import distutils.version

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.time = []
        self.meta = []

    def add(self, state, action, reward, terminal, features, 
                value = None, time = None, meta=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.terminal = terminal
        self.features += [features]
        if value is not None:
            self.values += [value]
        if time is not None:
            self.time += [time]
        if meta is not None:
            self.meta += [meta]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if other.values is not None:
            self.values.extend(other.values)
        if other.time is not None:
            self.time.extend(other.time)
        if other.meta is not None:
            self.meta.extend(other.meta)

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, solver):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.solver = solver
        self.num_local_steps = solver.t_max 
        self.env = solver.env
        self.last_features = None
        self.network = solver.local_network
        self.daemon = True
        self.sess = None
        self.summary_writer = None

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.network, self.num_local_steps, 
              self.summary_writer, solver=self.solver)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, network, num_local_steps, summary_writer, solver=None):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = env.reset()
    last_features = network.get_initial_features()
    last_meta = env.meta()
    if solver.use_target_network():
        last_target_features = solver.target_network.get_initial_features()

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            value = None
            
            # choose an action from the policy
            if not hasattr(solver, 'epsilon') or solver.epsilon() < np.random.uniform():
                fetched = network.act(last_state, last_features,
                        meta=last_meta)
                if network.type == 'policy':
                    action, value, features = fetched[0], fetched[1], fetched[2:]
                else:
                    action, features = fetched[0], fetched[1:]
            else: 
                # choose a random action
                assert network.type != 'policy'
                act_idx = np.random.randint(0, env.action_space.n)
                action = np.zeros(env.action_space.n)
                action[act_idx] = 1
                if network.is_recurrent():
                    features = network.update_state(last_state, last_features,
                            meta=last_meta)
                else:
                    features = []

            # argmax to convert from one-hot
            state, reward, terminal, info, time = env.step(action.argmax())
            if hasattr(env, 'atari'):
                reward = np.clip(reward, -1, 1)

            # collect the experience
            rollout.add(last_state, action, reward, terminal, last_features, 
                        value = value, time = time, meta=last_meta)

            last_state = state
            last_features = features
            last_meta = env.meta()

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, network.global_step.eval())
                summary_writer.flush()

            if terminal:
                terminal_end = True
                last_state = env.reset()
                last_features = network.get_initial_features()
                last_meta = env.meta()
                break

        if not terminal_end:
            if solver.use_target_network(): 
                rollout.r = solver.target_network.value(last_state, 
                            last_features,
                            meta=last_meta)
            else:
                rollout.r = network.value(last_state, last_features,
                            meta=last_meta)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class AsyncSolver(object):
    def __init__(self, env, args, env_off=None):
        self.env = env
        self.args = args
        self.task = args.task
        self.t_max = args.t_max
        self.ld = args.ld
        self.lr = args.lr
        self.model = args.model
        self.env_off = env_off
        self.last_global_step = 0

        device = 'gpu' if self.args.gpu > 0 else 'cpu'
        worker_device = "/job:worker/task:{}/{}:0".format(self.task, device)
        def _load_fn(unused_op):
            return 1
        with tf.device(tf.train.replica_device_setter(self.args.num_ps, 
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                    self.args.num_ps, _load_fn))):
            with tf.variable_scope("global"):
                with tf.variable_scope("learner"):
                    self.network = self.define_network(self.model) 
                if self.use_target_network():
                    with tf.variable_scope("target"):
                        self.global_target_network = self.define_network(self.model)
                    self.global_target_sync_step = tf.get_variable("target_sync_step", [], 
                            tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                            trainable=False)
                self.global_step = tf.get_variable("global_step", [], tf.int32, 
                            initializer=tf.constant_initializer(0, dtype=tf.int32),
                            trainable=False)


        with tf.device(worker_device):
            with tf.variable_scope("local"):
                with tf.variable_scope("learner"):
                    self.local_network = pi = self.define_network(self.model) 
                pi.global_step = self.global_step
                if self.use_target_network():
                    with tf.variable_scope("target"):
                        self.target_network = self.define_network(self.model)
            
            self.init_variables()

          # 20 represents the number of "local steps":  the number of timesteps
          # we run the policy before we update the parameters.
          # The larger local steps is, the lower is the variance in our policy gradients estimate
          # on the one hand;  but on the other hand, we get less frequent parameter updates, which
          # slows down learning.  In this code, we found that making local steps be much
          # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(self)

            self.grads = tf.gradients(self.loss, pi.var_list)
            self.grads, _ = tf.clip_by_global_norm(self.grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, 
                      self.network.var_list)])

            self.grads_and_vars = list(zip(self.grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = tf.group(opt.apply_gradients(self.grads_and_vars), inc_step)
            if self.use_target_network():
                self.update_target_step = self.global_target_sync_step.assign(self.global_step)

        with tf.device(None):
            self.define_summary()
            self.summary_writer = None
            self.local_steps = 0
    
    def define_summary(self):
        tf.summary.scalar("model/lr", self.learning_rate)
        tf.summary.image("model/state", self.env.tf_visualize(self.local_network.x), max_outputs=10)
        tf.summary.scalar("gradient/grad_norm", tf.global_norm(self.grads))
        tf.summary.scalar("param/param_norm", tf.global_norm(self.local_network.var_list))
        for grad_var in self.grads_and_vars:
            grad = grad_var[0]
            var = grad_var[1]
            if var.name.find('/W:') >= 0 or var.name.find('/w:') >= 0:
                if grad is None:
                    raise ValueError(var.name + " grads are missing")
                tf.summary.scalar("gradient/%s" % var.name, tf.norm(grad))
                tf.summary.scalar("param/%s" % var.name, tf.norm(var))

        self.summary_op = tf.summary.merge_all()

    def use_target_network(self):
        return False

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        '''
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        '''
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        should_compute_summary = self.task == 0 and self.local_steps % 101 == 0
        
        if self.local_steps % self.args.update_freq == 0:
            batch = self.process_rollout(rollout, gamma=self.args.gamma, lambda_=self.ld)
            extra_fetches = self.extra_fetches()
            if should_compute_summary:
                fetches = [self.train_op, self.summary_op, self.global_step]
            else:
                fetches = [self.train_op, self.global_step]

            feed_dict = self.prepare_input(batch)
            feed_dict[self.learning_rate] = \
                    self.args.lr * self.args.decay ** (self.last_global_step/float(10**6))
            fetched = sess.run(extra_fetches + fetches, feed_dict=feed_dict)
            if should_compute_summary:
                self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
                self.write_extra_summary(rollout=rollout)
                self.summary_writer.flush()
            self.last_global_step = fetched[-1]
            self.handle_extra_fetches(fetched[:len(extra_fetches)])

        self.local_steps += 1
        self.post_process(sess)
  
    def extra_fetches(self):
        return []

    def handle_extra_fetches(self, fetches):
        return None

    def post_process(self, sess):
        return None

    def write_extra_summary(self, rollout=None):
        return None
