from __future__ import print_function
import numpy as np
import tensorflow as tf
import model
import util
from async import AsyncSolver
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Q(AsyncSolver):
    def define_network(self, name):
        self.args.meta_dim = 0 if self.env.meta() is None else len(self.env.meta())
        return eval("model." + name)(self.env.observation_space.shape, 
                self.env.action_space.n, type='q',
                gamma=self.args.gamma, 
                dim=self.args.dim,
                f_num=self.args.f_num,
                f_pad=self.args.f_pad,
                f_stride=self.args.f_stride,
                f_size=self.args.f_size,
                meta_dim=self.args.meta_dim,
                )

    def use_target_network(self):
        return True

    def process_rollout(self, rollout, gamma, lambda_=1.0):
        """
    given a rollout, compute its returns
    """
        batch_si = np.asarray(rollout.states)
        batch_a = np.asarray(rollout.actions)
        rewards = np.asarray(rollout.rewards)
        time = np.asarray(rollout.time)
        meta = np.asarray(rollout.meta)
        rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
        batch_r = util.discount(rewards_plus_v, gamma, time)[:-1]
        features = rollout.features[0]

        return util.Batch(si=batch_si, 
              a=batch_a, 
              adv=None,
              r=batch_r, 
              terminal=rollout.terminal,
              features=features,
              reward=rewards,
              step=time,
              meta=meta,
              )

    def init_variables(self):
        pi = self.local_network
        
        # target network is synchronized after every 10,000 steps
        self.local_to_target = tf.group(*[v1.assign(v2) for v1, v2 in 
                        zip(self.global_target_network.var_list, pi.var_list)])
        self.target_sync = tf.group(*[v1.assign(v2) for v1, v2 in 
                        zip(self.target_network.var_list, self.global_target_network.var_list)])
        self.sync_count = 0
        self.target_sync_step = 0
        
        # epsilon 
        self.eps = [1.0]
        self.eps_start = [1.0]
        self.eps_end = [self.args.eps]
        self.eps_prob = [1]
        self.anneal_step = self.args.eps_step

        # batch size
        self.bs = tf.to_float(tf.shape(pi.x)[0])

        # loss function
        self.define_loss()

    def define_loss(self):
        pi = self.local_network
        
        # loss function
        self.ac = tf.placeholder(tf.float32, [None, self.env.action_space.n], name="ac")
        self.r = tf.placeholder(tf.float32, [None], name="r") # target 
        
        self.q_val = tf.reduce_sum(pi.q * self.ac, [1])
        self.delta = self.q_val - self.r
        # clipping gradient to [-1, 1] amounts to using Huber loss
        self.q_loss = tf.reduce_sum(tf.where(tf.abs(self.delta) < 1,
                                0.5 * tf.square(self.delta),
                                tf.abs(self.delta) - 0.5))

        self.loss = self.q_loss

    def define_summary(self):
        super(Q, self).define_summary()
        tf.summary.scalar("loss/loss", self.loss / self.bs)
        if hasattr(self, "q_loss"):
            tf.summary.scalar("loss/q_loss", self.q_loss / self.bs)
        tf.summary.scalar("param/target_param_norm",
              tf.global_norm(self.target_network.var_list))
        self.summary_op = tf.summary.merge_all()

    def start(self, sess, summary_writer):
        sess.run(self.sync) # copy weights from shared to local
        if self.task == 0: 
            sess.run(self.local_to_target) # copy weights from local to shared target 
        sess.run(self.target_sync) # copy weights from global target to local target
        super(Q, self).start(sess, summary_writer)
    
    def prepare_input(self, batch):
        feed_dict = {self.local_network.x: batch.si,
            self.ac: batch.a,
            self.r: batch.r}
        if self.args.meta_dim > 0:
            feed_dict[self.local_network.meta] = batch.meta
        for i in range(len(self.local_network.state_in)):
            feed_dict[self.local_network.state_in[i]] = batch.features[i]
        return feed_dict

    def post_process(self, sess):
        if self.task == 0:
            global_step = self.last_global_step
            if int(global_step / self.args.sync) > self.sync_count:
                # copy weights from local to shared target 
                self.sync_count = int(global_step / self.args.sync)
                sess.run([self.local_to_target, self.target_sync, self.update_target_step]) 
                logger.info("[Step: %d] Target network is synchronized", global_step)
        else:
            target_step = self.global_target_sync_step.eval()
            if target_step != self.target_sync_step:
                self.target_sync_step = target_step
                sess.run(self.target_sync)
                logger.info("[Step: %d] Target network is synchronized", target_step)

        for i in range(len(self.eps)):
            self.eps[i] = self.eps_start[i]
            self.eps[i] -= self.last_global_step * (self.eps_start[i] - self.eps_end[i])\
                    / self.anneal_step
            self.eps[i] = max(self.eps[i], self.eps_end[i])

    def epsilon(self):
        return np.random.choice(self.eps, p=self.eps_prob)

    def write_extra_summary(self, rollout=None):
        summary = tf.Summary()
        summary.value.add(tag='model/epsilon', simple_value=float(
              np.sum(np.array(self.eps) * np.array(self.eps_prob))))
        summary.value.add(tag='model/rollout_r', simple_value=float(rollout.r))
        self.summary_writer.add_summary(summary, self.last_global_step)
