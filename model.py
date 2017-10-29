import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import math

act_fn = tf.nn.elu

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def he_initializer(fan_in, uniform=False, factor=2, seed=None):
    def _initializer(shape, dtype=None, partition_info=None):
        n = fan_in
        if uniform:
          # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
          limit = math.sqrt(3.0 * factor / n)
          return tf.random_uniform(shape, -limit, limit, dtype, seed=seed)
        else:
          # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
          trunc_stddev = math.sqrt(1.3 * factor / n)
          return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=seed)
    return _initializer

def flatten(x):
    if x.get_shape().ndims > 2:
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
    return x

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None, init="he"):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        if init != "he":
            fan_in = np.prod(filter_shape[:3])
            fan_out = np.prod(filter_shape[:2]) * num_filters
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            w = tf.get_variable("W", filter_shape, dtype, 
                    tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
        else:
            fan_in = np.prod(filter_shape[:3])
            w = tf.get_variable("W", filter_shape, dtype, he_initializer(fan_in),
                                collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], 
                initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

def transform_fc(x, a, n_actions, name, bias_init=0, pad="SAME"):
    if x.shape.ndims > 2:
        x = flatten(x)
    xdim = int(x.get_shape()[1])
    w = tf.get_variable(name + "/w", [n_actions, xdim], 
            initializer=normalized_columns_initializer(0.1))
    if a is not None: 
        # Transform only for the given action
        mul = x * w[tf.to_int32(tf.squeeze(tf.argmax(a, axis=1))), :] 
    else:
        # Enumerate all possible actions and concatenate them
        transformed = []
        for i in range(0, n_actions):
            transformed.append(x * w[i, :])
        mul = pack(tf.concat(transformed, 1), [xdim])

    h = linear(mul, xdim, name + "_dec", 
        initializer=normalized_columns_initializer(0.1), bias_init=bias_init)
    return act_fn(h)

def transform_conv_state(x, a, n_actions, filter_size=(3, 3), pad="SAME"):
    # 3x3 option-conv -> 3x3 conv * 1x1 mask (with residual connection)
    stride_shape = [1, 1, 1, 1]
    dec_f_size = filter_size[0]
    num_filters = int(x.get_shape()[3])
    xdim = [int(x.get_shape()[1]), int(x.get_shape()[2]), num_filters]
    filter_shape = [filter_size[0], filter_size[1], num_filters, n_actions, num_filters]
    fan_in = np.prod(filter_shape[:3])
    dec_filter_shape = [dec_f_size, dec_f_size, num_filters, num_filters]
    w = tf.get_variable("W", filter_shape, initializer=he_initializer(fan_in))
    b = tf.get_variable("b", [1, 1, 1, n_actions, num_filters], 
                    initializer=tf.constant_initializer(0.0))
    w_dec = tf.get_variable("dec1-W", dec_filter_shape, 
                    initializer=he_initializer(fan_in))
    b_dec = tf.get_variable("dec1-b", [1, 1, 1, num_filters], 
                    initializer=tf.constant_initializer(0.0))
    w_dec2 = tf.get_variable("dec2-W", dec_filter_shape, 
                    initializer=he_initializer(fan_in))
    b_dec2 = tf.get_variable("dec2-b", [1, 1, 1, num_filters], 
                    initializer=tf.constant_initializer(0.0))
    w_gate = tf.get_variable("gate-W", [1, 1, num_filters, num_filters], 
                    initializer=he_initializer(num_filters))
    b_gate = tf.get_variable("gate-b", [1, 1, 1, num_filters], 
                    initializer=tf.constant_initializer(0.0))
    if a is not None:
        idx = tf.to_int32(tf.squeeze(tf.argmax(a, axis=1)))
        conv = tf.nn.conv2d(x, w[:, :, :, idx, :], stride_shape, pad) + b[:, :, :, idx, :]
        conv = act_fn(conv)
    else:
        w = tf.reshape(w, [filter_size[0], filter_size[1], num_filters, n_actions * num_filters]) 
        b = tf.reshape(b, [1, 1, 1, n_actions * num_filters])
        conv = act_fn(tf.nn.conv2d(x, w, stride_shape, pad) + b)
        conv = pack(tf.transpose(tf.reshape(conv, 
                    [-1, xdim[0], xdim[1], n_actions, num_filters]),
                    [0, 3, 1, 2, 4]), xdim)
    conv = act_fn(tf.nn.conv2d(conv, w_dec, stride_shape, pad) + b_dec)
    gate = tf.sigmoid(tf.nn.conv2d(conv, w_gate, stride_shape, pad) + b_gate)
    conv = tf.nn.conv2d(conv, w_dec2, stride_shape, pad) + b_dec2 
    if a is not None:
        conv = conv * gate + x
    else:
        conv = tf.transpose(tf.reshape(conv, [-1, n_actions] + xdim), [1, 0, 2, 3, 4])
        gate = tf.transpose(tf.reshape(gate, [-1, n_actions] + xdim), [1, 0, 2, 3, 4])
        conv = conv * gate + x
        conv = pack(tf.transpose(conv, [1, 0, 2, 3, 4]), xdim)
    return act_fn(conv)

def transform_conv_pred(x, a, n_actions, filter_size=(3, 3), pad="SAME"):
    # 3x3 option-conv -> 3x3 conv
    stride_shape = [1, 1, 1, 1]
    dec_f_size = filter_size[0]
    num_filters = int(x.get_shape()[3])
    xdim = [int(x.get_shape()[1]), int(x.get_shape()[2]), num_filters]
    filter_shape = [filter_size[0], filter_size[1], num_filters, n_actions, num_filters]
    fan_in = np.prod(filter_shape[:3])
    fan_out = np.prod(filter_shape[:2]) * num_filters
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    w = tf.get_variable("W", filter_shape, 
                    initializer=tf.random_uniform_initializer(-w_bound, w_bound))
    b = tf.get_variable("b", [1, 1, 1, n_actions, num_filters], 
                    initializer=tf.constant_initializer(0.0))
    w_dec = tf.get_variable("W-dec", [dec_f_size, dec_f_size, num_filters, num_filters], 
                    initializer=tf.random_uniform_initializer(-w_bound, w_bound))
    b_dec = tf.get_variable("b-dec", [1, 1, 1, num_filters], 
                    initializer=tf.constant_initializer(0.0))
    if a is not None:
        idx = tf.to_int32(tf.squeeze(tf.argmax(a, axis=1)))
        conv = tf.nn.conv2d(x, w[:, :, :, idx, :], stride_shape, pad) + b[:, :, :, idx, :]
        conv = act_fn(conv)
    else:
        w = tf.reshape(w, [filter_size[0], filter_size[1], num_filters, n_actions * num_filters]) 
        b = tf.reshape(b, [1, 1, 1, n_actions * num_filters])
        conv = act_fn(tf.nn.conv2d(x, w, stride_shape, pad) + b)
        conv = pack(tf.transpose(tf.reshape(conv, 
                    [-1, xdim[0], xdim[1], n_actions, num_filters]),
                    [0, 3, 1, 2, 4]), xdim)
    conv = tf.nn.conv2d(conv, w_dec, stride_shape, pad) + b_dec 
    return act_fn(conv)
    
def pack(x, dim):
    return tf.reshape(x, [-1] + dim)

def to_value(x, dim=256, initializer=None, bias_init=0):
    if x.shape.ndims == 2: # fc layer
        return linear(x, 1, "v", initializer=initializer, bias_init=bias_init)
    else: # conv layer
        x = act_fn(linear(flatten(x), dim, "v1", 
            initializer=tf.contrib.layers.xavier_initializer(), bias_init=bias_init))
        return linear(x, 1, "v", initializer=initializer, bias_init=bias_init)

def to_pred(x, dim=256, initializer=None, bias_init=0):
    return linear(flatten(x), dim, "p", initializer=initializer, bias_init=bias_init)

def to_reward(x, dim=256, initializer=None, bias_init=0):
    x = act_fn(linear(flatten(x), dim, "r1", 
            initializer=tf.contrib.layers.xavier_initializer(), bias_init=bias_init))
    return linear(x, 1, "r", initializer=initializer, bias_init=bias_init)

def to_steps(x, dim=256, initializer=None, bias_init=0):
    x = act_fn(linear(flatten(x), dim, "t1", 
            initializer=tf.contrib.layers.xavier_initializer(), bias_init=bias_init))
    return linear(x, 1, "t", initializer=initializer, bias_init=bias_init)

def rollout_step(x, a, n_actions, op_trans_state, op_trans_pred,
                op_value, op_steps, op_reward, gamma=0.98):
    state = op_trans_state(x, a)
    p = op_trans_pred(x, a)
    if a is not None:
        v_next = op_value(state)
        r = op_reward(p)
        t = op_steps(p)
    else:
        v_next = pack(op_value(state), [n_actions])
        r = pack(op_reward(p), [n_actions])
        t = pack(op_steps(p), [n_actions])
   
    t = tf.nn.relu(t) + 1
    g = tf.pow(tf.constant(gamma), t)
    return r, g, t, v_next, state

def predict_over_time(x, a, n_actions, op_rollout, prediction_step=5):
    time_steps = tf.shape(a)[0]
    xdim = x.get_shape().as_list()[1:]
    
    def _create_ta(name, dtype, size, clear=True):
        return tf.TensorArray(dtype=dtype, 
                    size=size, tensor_array_name=name,
                    clear_after_read=clear)
    
    v_ta = _create_ta("output_v", x.dtype, time_steps) 
    r_ta = _create_ta("output_r", x.dtype, time_steps)
    g_ta = _create_ta("output_g", x.dtype, time_steps)
    t_ta = _create_ta("output_t", x.dtype, time_steps)
    q_ta = _create_ta("output_q", x.dtype, time_steps)
    s_ta = _create_ta("output_s", x.dtype, time_steps)

    x_ta = _create_ta("input_x", x.dtype, time_steps).unstack(x)
    a_ta = _create_ta("input_a", x.dtype, time_steps).unstack(a)
    
    time = tf.constant(0, dtype=tf.int32)
    roll_step = tf.minimum(prediction_step, time_steps)
    state = tf.zeros([roll_step] + xdim)
    
    def _time_step(time, r_ta, g_ta, t_ta, v_ta, q_ta, s_ta, state):
        a_t = a_ta.read(time)
        a_t = tf.expand_dims(a_t, 0)

        # stack previously generated states with the new state through batch
        x_t = x_ta.read(time)
        x_t = tf.expand_dims(x_t, 0)
        state = tf.concat([x_t, tf.slice(state, [0] * (len(xdim) + 1), 
                    [roll_step-1] + xdim)], 0)

        r, gamma, t, v_next, state = op_rollout(state, a_t)
        q = r + gamma * v_next
        r_ta = r_ta.write(time, tf.reshape(r, [-1]))
        g_ta = g_ta.write(time, tf.reshape(gamma, [-1]))
        t_ta = t_ta.write(time, tf.reshape(t, [-1]))
        v_ta = v_ta.write(time, tf.reshape(v_next, [-1]))
        q_ta = q_ta.write(time, tf.reshape(q, [-1]))
        s_ta = s_ta.write(time, state)
        return (time+1, r_ta, g_ta, t_ta, v_ta, q_ta, s_ta, state)

    _, r_ta, g_ta, t_ta, v_ta, q_ta, s_ta, state = tf.while_loop(
                          cond=lambda time, *_: time < time_steps,
                          body=_time_step,
                          loop_vars=(time, v_ta, r_ta, g_ta, t_ta, q_ta, s_ta, state))

    r = r_ta.stack()
    g = g_ta.stack()
    t = t_ta.stack()
    v = v_ta.stack()
    q = q_ta.stack()
    s = s_ta.stack()

    return r, g, t, v, q, s

class Model(object):
    def __init__(self, ob_space, n_actions, type, 
                    gamma=0.99, prediction_step=1,
                    dim=256,
                    f_num=[32,32,64],
                    f_stride=[1,1,2],
                    f_size=[3,3,4],
                    f_pad="SAME",
                    branch=[4,4,4],
                    meta_dim=0):
        self.n_actions = n_actions
        self.type = type
        self.x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.a = tf.placeholder(tf.float32, [None, n_actions])
        self.meta = tf.placeholder(tf.float32, [None, meta_dim]) if meta_dim > 0 else None
        self.state_init = []
        self.state_in = []
        self.state_out = []
        self.dim = dim
        self.f_num = f_num
        self.f_stride = f_stride
        self.f_size = f_size
        self.f_pad = f_pad
        self.meta_dim = meta_dim
        self.xdim = list(ob_space)
        self.branch = [min(n_actions, k) for k in branch]

        self.s, self.state_in, self.state_out = self.build_model(self.x, self.meta)
        self.sdim = self.s.get_shape().as_list()[1:]

        # output layer
        if self.type == 'policy':
            self.logits = linear(flatten(self.s), n_actions, "action", 
                    normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(flatten(self.s), 1, "value", 
                normalized_columns_initializer(1.0)), [-1])
            self.sample = categorical_sample(self.logits, n_actions)[0, :]
        elif self.type == 'q':
            h = transform_conv_state(self.s, None, n_actions)
            self.h = linear(flatten(h), self.dim, "fc", 
                            normalized_columns_initializer(0.01)) 
            self.q = pack(linear(self.h, 1, "action", 
                    normalized_columns_initializer(0.01)), [n_actions])
            self.sample = tf.one_hot(tf.squeeze(tf.argmax(self.q, axis=1)), n_actions)
            self.qmax = tf.reduce_max(self.q, axis=[1])
        elif self.type == 'vpn':
            self.op_value = tf.make_template('v', to_value, dim=self.dim,
                            initializer=normalized_columns_initializer(0.01))
            self.op_reward = tf.make_template('r', to_reward, dim=self.dim,
                    initializer=normalized_columns_initializer(0.01))
            self.op_steps = tf.make_template('t', to_steps, dim=self.dim,
                    initializer=normalized_columns_initializer(0.01))
            self.op_trans_state = tf.make_template('trans_state', transform_conv_state, 
                    n_actions=n_actions)
            self.op_trans_pred = tf.make_template('trans_pred', transform_conv_pred, 
                    n_actions=n_actions)
            self.op_rollout = tf.make_template('rollout', rollout_step, 
                    n_actions=n_actions,
                    op_trans_state=self.op_trans_state, 
                    op_trans_pred=self.op_trans_pred,
                    op_value=self.op_value, 
                    op_steps=self.op_steps, 
                    op_reward=self.op_reward,
                    gamma=gamma)

            # Unconditional rollout
            self.r, self.gamma, self.steps, self.v_next, self.state = self.op_rollout(self.s, None)
            self.q = self.r + self.gamma * self.v_next

            # Action-conditional rollout over time for training
            self.r_a, self.gamma_a, self.t_a, self.v_next_a, self.q_a, self.states = \
                    predict_over_time(self.s, self.a, n_actions, self.op_rollout,
                            prediction_step=prediction_step)
          
            # Tree expansion/backup
            depth = len(self.branch)
            q_list = []
            r_list = []
            g_list = []
            v_list = []
            idx_list = []
            s_list = []
            s = self.s
            
            # Expansion
            for i in range(depth):
                r, gamma, _, v, s = self.op_rollout(s, None)
                r_list.append(tf.squeeze(r))
                v_list.append(tf.squeeze(v))
                s_list.append(s)
                g_list.append(tf.squeeze(gamma))

                b = self.branch[i] 
                q_list.append(r_list[i] + g_list[i] * v_list[i])
                q_list[i] = tf.reshape(q_list[i], [-1, self.n_actions])
                _, idx = tf.nn.top_k(q_list[i], k=b)
                idx_list.append(idx)

                l = tf.tile(tf.expand_dims(tf.range(0, tf.shape(idx)[0]), 1), [1, b])
                l = tf.concat([tf.reshape(l, [-1, 1]), tf.reshape(idx, [-1, 1])], axis=1)
                s = tf.reshape(tf.gather_nd(
                        tf.reshape(s, [-1, self.n_actions] + self.sdim), l), [-1] + self.sdim)
                r_list[i] = tf.reshape(tf.gather_nd(
                        tf.reshape(r_list[i], [-1, self.n_actions]), l), [-1])
                g_list[i] = tf.reshape(tf.gather_nd(
                        tf.reshape(g_list[i], [-1, self.n_actions]), l), [-1])
                v_list[i] = tf.reshape(tf.gather_nd(
                        tf.reshape(v_list[i], [-1, self.n_actions]), l), [-1])

            self.q_list = q_list
            self.r_list = r_list
            self.g_list = g_list
            self.v_list = v_list
            self.s_list = s_list
            self.idx_list = idx_list

            # Backup
            v_plan = [None] * depth
            q_plan = [None] * depth

            v_plan[-1] = v_list[-1]
            for i in reversed(range(0, depth)):
                q_plan[i] = r_list[i] + g_list[i] * v_plan[i]
                if i > 0:
                    q_max = tf.reduce_max(tf.reshape(q_plan[i], [-1, self.branch[i]]), axis=1)
                    n = float(depth - i)
                    v_plan[i-1] = (v_list[i-1] + q_max * n) / (n + 1)

            idx = tf.squeeze(idx_list[0])
            self.q_deep = tf.squeeze(q_plan[0])
            self.q_plan = tf.sparse_to_dense(idx, [self.n_actions], self.q_deep, 
                        default_value=-100, validate_indices=False)

            self.x_off = tf.placeholder(tf.float32, [None] + list(ob_space))
            self.a_off = tf.placeholder(tf.float32, [None, n_actions])
            self.meta_off = tf.placeholder(tf.float32, [None, self.meta_dim]) \
                    if self.meta_dim > 0 else None
            tf.get_variable_scope().reuse_variables()
            self.s_off, self.state_in_off, self.state_out_off = \
                    self.build_model(self.x_off, self.meta_off)
            
            # Action-conditional rollout over time for training
            self.r_off, self.gamma_off, self.t_off, self.v_next_off, _, _ = \
                predict_over_time(self.s_off, self.a_off, n_actions, self.op_rollout,
                        prediction_step=prediction_step)
            
        else:
            raise ValueError('Invalid model type %s' % (self.type))
            
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            tf.get_variable_scope().name)

        self.num_param = 0
        for v in self.var_list:
            self.num_param += v.get_shape().num_elements()
   
    def is_recurrent(self):
        return self.state_in is not None and len(self.state_in) > 0

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, state_in=[], meta=None):
        sess = tf.get_default_session()
        feed_dict = {self.x: [ob]}
        for i in range(len(state_in)):
            feed_dict[self.state_in[i]] = state_in[i]
        if self.meta_dim > 0:
            feed_dict[self.meta] = [meta]

        if self.type == 'policy':
            return sess.run([self.sample, self.vf] + self.state_out, feed_dict)
        elif self.type == 'q':
            return sess.run([self.sample] + self.state_out, feed_dict)
        elif self.type == 'vpn':
            out = sess.run([self.q_plan] + self.state_out, feed_dict)
            q = out[0]
            state_out = out[1:]
            act = np.zeros_like(q)
            act[q.argmax()] = 1
            return [act] + state_out

    def update_state(self, ob, state_in=[], meta=None):
        sess = tf.get_default_session()
        feed_dict = {self.x: [ob]}
        for i in range(len(state_in)):
            feed_dict[self.state_in[i]] = state_in[i]
        if self.meta_dim > 0:
            feed_dict[self.meta] = [meta]
        return sess.run(self.state_out, feed_dict)

    def value(self, ob, state_in=[], meta=None):
        sess = tf.get_default_session()
        feed_dict = {self.x: [ob]}
        for i in range(len(state_in)):
            feed_dict[self.state_in[i]] = state_in[i]
        if self.meta_dim > 0:
            feed_dict[self.meta] = [meta]

        if self.type == 'policy':
            return sess.run(self.vf, feed_dict)[0]
        elif self.type == 'q':
            return sess.run(self.qmax, feed_dict)[0]
        elif self.type == 'vpn':
            q = sess.run(self.q_plan, feed_dict)
            return q.max()

class CNN(Model):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)

    def build_model(self, x, meta=None):
        for i in range(len(self.f_num)):
            x = act_fn(conv2d(x, self.f_num[i], "l{}".format(i+1), 
                        [self.f_size[i], self.f_size[i]], 
                        [self.f_stride[i], self.f_stride[i]], pad=self.f_pad, 
                        init="he"))
            self.conv = x
        if meta is not None:
            space_dim = x.get_shape().as_list()[1:3]
            meta_dim = meta.get_shape().as_list()[-1]
            t = tf.reshape(tf.tile(meta, 
                        [1, np.prod(space_dim)]), [-1] + space_dim + [meta_dim])
            x = tf.concat([t, x], axis=3)

        return x, [], []

class LSTM(Model):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
    
    def build_model(self, x):
        for i in range(len(self.f_num)):
            x = act_fn(conv2d(x, self.f_num[i], "l{}".format(i+1), 
                        [self.f_size[i], self.f_size[i]], 
                        [self.f_stride[i], self.f_stride[i]], pad=self.f_pad))
            self.conv = x
        x = act_fn(linear(flatten(x), 256, "l{}".format(3), 
              normalized_columns_initializer(0.01)))
        
        # introduce a "fake" batch dimension of 1 after flatten 
        # so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        
        x = tf.reshape(lstm_outputs, [-1, size])
        return x, state_in, state_out
