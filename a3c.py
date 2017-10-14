from __future__ import print_function
import numpy as np
import tensorflow as tf
import util
from async import AsyncSolver
import model

class A3C(AsyncSolver):
    def define_network(self, name):
        self.args.meta_dim = 0 if self.env.meta() is None else len(self.env.meta())
        return eval("model." + name)(self.env.observation_space.shape, 
                self.env.action_space.n, type='policy',
                gamma=self.args.gamma, 
                dim=self.args.dim,
                f_num=self.args.f_num,
                f_pad=self.args.f_pad,
                f_stride=self.args.f_stride,
                f_size=self.args.f_size,
                meta_dim=self.args.meta_dim,
                )

    def process_rollout(self, rollout, gamma, lambda_=1.0):
        """
    given a rollout, compute its returns and the advantage
    """
        batch_si = np.asarray(rollout.states)
        batch_a = np.asarray(rollout.actions)
        rewards = np.asarray(rollout.rewards)
        time = np.asarray(rollout.time)
        meta = np.asarray(rollout.meta)
        vpred_t = np.asarray(rollout.values + [rollout.r])

        rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
        batch_r = util.discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        batch_adv = util.discount(delta_t, gamma * lambda_)

        features = rollout.features[0]
        return util.Batch(si=batch_si, 
                a=batch_a, 
                adv=batch_adv, 
                r=batch_r, 
                terminal=rollout.terminal, 
                features=features,
                reward=rewards,
                step=time,
                meta=meta)

    def init_variables(self):
        pi = self.local_network
        self.ac = tf.placeholder(tf.float32, [None, self.env.action_space.n], name="ac")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        log_prob_tf = tf.nn.log_softmax(pi.logits)
        prob_tf = tf.nn.softmax(pi.logits)

        # the "policy gradients" loss:  its derivative is precisely the policy gradient
        # notice that self.ac is a placeholder that is provided externally.
        # adv will contain the advantages, as calculated in process_rollout
        self.pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

        # loss of value function
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
        self.entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

        self.bs = tf.to_float(tf.shape(pi.x)[0])
        self.loss = self.pi_loss + 0.5 * self.vf_loss - self.entropy * 0.01

    def define_summary(self):
        super(A3C, self).define_summary()
        tf.summary.scalar("model/policy_loss", self.pi_loss / self.bs)
        tf.summary.scalar("model/value_loss", self.vf_loss / self.bs)
        tf.summary.scalar("model/entropy", self.entropy / self.bs)
        self.summary_op = tf.summary.merge_all()
    
    def prepare_input(self, batch):
        feed_dict = {self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r}
        if self.args.meta_dim > 0:
            feed_dict[self.local_network.meta] = batch.meta
        for i in range(len(self.local_network.state_in)):
            feed_dict[self.local_network.state_in[i]] = batch.features[i]
        return feed_dict
