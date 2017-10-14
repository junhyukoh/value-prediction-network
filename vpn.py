from __future__ import print_function
import numpy as np
import tensorflow as tf
import model  # NOQA
import util
from q import Q
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RolloutMemory(object):
    def __init__(self, max_size, sampling='rand'):
        self.max_size = max_size
        self.s = []
        self.a = []
        self.r = []
        self.t = []
        self.r_t = []
        self.term = []
        self.sampling = sampling
        self.sample_idx = 0

    def add(self, s, a, r, t, r_t, term):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.t.append(t)
        self.r_t.append(r_t)
        self.term.append(term)

    def size(self):
        return len(self.s)

    def is_full(self):
        return len(self.s) >= self.max_size

    def sample(self, length):
        size = len(self.s)
        is_initial_state = False
        if self.sampling == 'rand':
            idx = np.random.randint(0, size-1)
            if self.term[idx]:
                return self.sample(length)
            for end_idx in range(idx, idx + length):
                if self.term[end_idx] or end_idx == size-1:
                    break
            is_initial_state = (idx > 0 and self.term[idx-1]) or idx == 0
        else:
            idx = self.sample_idx
            if self.term[idx]:
                idx = idx + 1
            for end_idx in range(idx, idx + length):
                if self.term[end_idx] or end_idx == size-1:
                    break
            self.sample_idx = end_idx + 1 if end_idx < size-1 else 0
            is_initial_state = (idx > 0 and self.term[idx-1]) or idx == 0

        assert end_idx == idx + length - 1 or self.term[end_idx] or end_idx == size-1
        return util.Batch(si=np.asarray(self.s[idx:end_idx+1]), 
              a=np.asarray(self.a[idx:end_idx+1]), 
              adv=None,
              r=None, 
              terminal=self.term,
              features=[],
              reward=np.asarray(self.r[idx:end_idx+1]),
              step=np.asarray(self.t[idx:end_idx+1]),
              meta=np.asarray(self.r_t[idx:end_idx+1])), is_initial_state

class VPN(Q):
    def define_network(self, name):
        self.state_off = None
        self.args.meta_dim = 0 if self.env.meta() is None else len(self.env.meta())
        m = eval("model." + name)(self.env.observation_space.shape, 
                self.env.action_space.n, type='vpn', 
                gamma=self.args.gamma, 
                prediction_step=self.args.prediction_step,
                dim=self.args.dim,
                f_num=self.args.f_num,
                f_pad=self.args.f_pad,
                f_stride=self.args.f_stride,
                f_size=self.args.f_size,
                branch=self.args.branch,
                meta_dim=self.args.meta_dim,
                )

        return m
    
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
        batch_r = util.discount(rewards_plus_v, gamma, time)
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

    def define_loss(self):
        pi = self.local_network
        if self.args.buf > 0:
            if pi.is_recurrent():
                self.rand_rollouts = RolloutMemory(int(self.args.buf / self.args.num_workers),
                        sampling='seq')
                self.off_state = pi.get_initial_features()
            else:
                self.rand_rollouts = RolloutMemory(int(self.args.buf / self.args.num_workers))
        
        # loss function
        self.ac = tf.placeholder(tf.float32, [None, self.env.action_space.n], name="ac")
        self.v_target = tf.placeholder(tf.float32, [None], name="v_target") # target 
        self.reward = tf.placeholder(tf.float32, [None], name="reward") # immediate reward 
        self.step = tf.placeholder(tf.float32, [None], name="step") # num of steps
        self.terminal = tf.placeholder(tf.float32, (), name="terminal")
        
        time = tf.shape(pi.x)[0]
        steps = tf.minimum(self.args.prediction_step, time)
        self.rollout_num = tf.to_float(time * steps - steps * (steps - 1) / 2)
        
        # reward/gamma/value prediction
        self.r_delta = util.lower_triangular(
                    pi.r_a - tf.reshape(self.reward, [-1, 1]))
        self.r_loss_mat = util.huber_loss(self.r_delta, sum=False)
        self.r_loss = tf.reduce_sum(self.r_loss_mat)

        self.gamma_loss_mat = util.huber_loss(util.lower_triangular(
                pi.t_a - tf.reshape(self.step, [-1, 1])), sum=False)
        self.gamma_loss = tf.reduce_sum(self.gamma_loss_mat)

        self.v_next_loss_mat = util.huber_loss(util.lower_triangular(
            pi.v_next_a - tf.reshape(self.v_target[1:], [-1, 1])), sum=False)
        self.v_next_loss = tf.reduce_sum(self.v_next_loss_mat)
        self.loss = self.r_loss + self.gamma_loss + self.v_next_loss 

        # reward/gamma prediction for off-policy data (optional)
        self.a_off = tf.placeholder(tf.float32, [None, self.env.action_space.n], name="a_off")
        self.r_off = tf.placeholder(tf.float32, [None], name="r_off") # immediate reward 
        self.step_off = tf.placeholder(tf.float32, [None], name="step_off") # num of steps
        
        self.r_delta_off = util.lower_triangular(
                    pi.r_off - tf.reshape(self.r_off, [-1, 1]))
        self.r_loss_mat_off = util.huber_loss(self.r_delta_off, sum=False)
        self.r_loss_off = tf.reduce_sum(self.r_loss_mat_off)

        self.gamma_loss_mat_off = util.huber_loss(util.lower_triangular(
                pi.t_off - tf.reshape(self.step_off, [-1, 1])), sum=False)
        
        self.gamma_loss_off = tf.reduce_sum(self.gamma_loss_mat_off)
        self.loss += self.r_loss_off + self.gamma_loss_off 

    def prepare_input(self, batch):
        feed_dict = {self.local_network.x: batch.si,
            self.local_network.a: batch.a,
            self.ac: batch.a,
            self.reward: batch.reward,
            self.step: batch.step,
            self.target_network.x: batch.si,
            self.terminal: float(batch.terminal),
            self.v_target: batch.r}

        for i in range(len(self.local_network.state_in)):
            feed_dict[self.local_network.state_in[i]] = batch.features[i]

        if self.args.meta_dim > 0:
            feed_dict[self.local_network.meta] = batch.meta

        traj, initial = self.random_trajectory()
        feed_dict[self.local_network.x_off] = traj.si
        feed_dict[self.local_network.a_off] = traj.a
        feed_dict[self.a_off] = traj.a
        feed_dict[self.r_off] = traj.reward
        feed_dict[self.step_off] = traj.step
        
        if self.local_network.is_recurrent():
            if initial:
                state_in = self.local_network.get_initial_features()
            else:
                state_in = self.off_state
            for i in range(len(self.local_network.state_in_off)):
                feed_dict[self.local_network.state_in_off[i]] = state_in[i]

        if self.args.meta_dim > 0:
            feed_dict[self.local_network.meta_off] = traj.meta

        return feed_dict

    def random_trajectory(self):
        if not self.rand_rollouts.is_full():
            env = self.env_off
            state_off = env.reset()
            meta_off = env.meta()
            print("Generating random rollouts: %d steps" % self.rand_rollouts.max_size)
            while not self.rand_rollouts.is_full():
                act_idx = np.random.randint(0, env.action_space.n)
                action = np.zeros(env.action_space.n)
                action[act_idx] = 1
                state, reward, terminal, _, time = env.step(action.argmax())
                self.rand_rollouts.add(state_off, action, reward, time, 
                        meta_off, terminal)
                state_off = state
                meta_off = env.meta()
                if terminal:
                    state_off = env.reset()
                    meta_off = env.meta()
        return self.rand_rollouts.sample(self.args.t_max)

    def extra_fetches(self):
        if self.local_network.is_recurrent():
            return self.local_network.state_out_off
        return []

    def handle_extra_fetches(self, fetches):
        if self.local_network.is_recurrent():
            self.off_state = fetches[:len(self.off_state)]

    def compute_depth(self, steps):
        return self.args.depth

    def write_extra_summary(self, rollout=None):
        super(VPN, self).write_extra_summary(rollout)

    def define_summary(self):
        super(VPN, self).define_summary()
        tf.summary.scalar("loss/r_loss", self.r_loss / self.rollout_num)
        tf.summary.scalar("loss/gamma_loss", self.gamma_loss / self.rollout_num)
        tf.summary.scalar("model/r", tf.reduce_mean(self.local_network.r))
        tf.summary.scalar("model/v_next", tf.reduce_mean(self.local_network.v_next))
        tf.summary.scalar("model/gamma", tf.reduce_mean(self.local_network.gamma))
        self.summary_op = tf.summary.merge_all()
