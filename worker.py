#!/usr/bin/env python
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
from a3c import A3C
from q import Q
from vpn import VPN
from envs import create_env
import util
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def new_env(args):
    config = open(args.config) if args.config != "" else None
    env = create_env(args.env_id,
        str(args.task), 
        args.remotes,
        config=config)
    return env

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    env = new_env(args)
    if args.alg == 'A3C': 
        trainer = A3C(env, args)
    elif args.alg == 'Q':
        trainer = Q(env, args)
    elif args.alg == 'VPN':
        env_off = new_env(args)
        env_off.verbose = 0
        env_off.reset()
        trainer = VPN(env, args, env_off=env_off)
    else:
        raise ValueError('Invalid algorithm: ' + args.alg)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() if \
                not v.name.startswith("global") and not v.name.startswith("local/target/")]
    global_variables = [v for v in tf.global_variables() if not v.name.startswith("local")]

    init_op = tf.variables_initializer(global_variables)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save, max_to_keep=0)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())
    logger.info("Num parameters: %d", trainer.local_network.num_param)

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    device = 'gpu' if args.gpu > 0 else 'cpu'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    config = tf.ConfigProto(device_filters=["/job:ps", 
                "/job:worker/task:{}/{}:0".format(args.task, device)],
                gpu_options=gpu_options,
                allow_soft_placement=True)
    logdir = os.path.join(args.log, 'train')
    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(global_variables),
                             global_step=trainer.global_step,
                             save_model_secs=0,
                             save_summaries_secs=30)


    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        epoch = -1
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not args.max_step or global_step < args.max_step):
            if args.task == 0 and int(global_step / args.eval_freq) > epoch:
                epoch = int(global_step / args.eval_freq)
                filename = os.path.join(args.log, 'e%d' % (epoch))
                sv.saver.save(sess, filename)
                sv.saver.save(sess, os.path.join(args.log, 'latest'))
                print("Saved to: %s" % filename)
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)
        
        if args.task == 0 and int(global_step / args.eval_freq) > epoch:
            epoch = int(global_step / args.eval_freq)
            filename = os.path.join(args.log, 'e%d' % (epoch))
            sv.saver.save(sess, filename)
            sv.saver.save(sess, os.path.join(args.log, 'latest'))
            print("Saved to: %s" % filename)
    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers + 1):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    port += 1
    return cluster

def evaluate(env, network, num_play=3000, eps=0.0):
    for iter in range(0, num_play):
        last_state = env.reset()
        last_features = network.get_initial_features()
        last_meta = env.meta()
        while True:
            if eps == 0.0 or np.random.rand() > eps:
                fetched = network.act(last_state, last_features,
                        meta=last_meta)
                if network.type == 'policy':
                    action, features = fetched[0], fetched[2:]
                else:
                    action, features = fetched[0], fetched[1:]
            else:
                act_idx = np.random.randint(0, env.action_space.n)
                action = np.zeros(env.action_space.n)
                action[act_idx] = 1
                features = []

            state, reward, terminal, info, time = env.step(action.argmax())
            last_state = state
            last_features = features
            last_meta = env.meta()

            if terminal:
                break

    return env.reward_mean(num_play)

def run_tester(args, server):
    env = new_env(args)
    env.reset()
    env.max_history = args.eval_num
    if args.alg == 'A3C': 
        agent = A3C(env, args)
    elif args.alg == 'Q':
        agent = Q(env, args)
    elif args.alg == 'VPN':
        agent = VPN(env, args)
    else:
        raise ValueError('Invalid algorithm: ' + args.alg)

    device = 'gpu' if args.gpu > 0 else 'cpu'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    config = tf.ConfigProto(device_filters=["/job:ps", 
                "/job:worker/task:{}/{}:0".format(args.task, device)],
                gpu_options=gpu_options,
                allow_soft_placement=True)
    variables_to_save = [v for v in tf.global_variables() if \
                not v.name.startswith("global") and not v.name.startswith("local/target/")]
    global_variables = [v for v in tf.global_variables() if not v.name.startswith("local")]
    
    init_op = tf.variables_initializer(global_variables)
    init_all_op = tf.global_variables_initializer()
 
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())
    logger.info("Num parameters: %d", agent.local_network.num_param)

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)
    
    saver = FastSaver(variables_to_save, max_to_keep=0)
    sv = tf.train.Supervisor(is_chief=False,
                             global_step=agent.global_step,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             ready_op=tf.report_uninitialized_variables(global_variables),
                             saver=saver,
                             save_model_secs=0,
                             save_summaries_secs=0)
   
    best_reward = -10000
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        epoch = args.eval_epoch
        while args.eval_freq * epoch <= args.max_step:
            path = os.path.join(args.log, "e%d" % epoch)
            if not os.path.exists(path + ".index"):
                time.sleep(10)
                continue
            logger.info("Start evaluation (Epoch %d)", epoch)
            saver.restore(sess, path)
            np.random.seed(args.seed)
            reward = evaluate(env, agent.local_network, args.eval_num, eps=args.eps_eval)

            logfile = open(os.path.join(args.log, "eval.csv"), "a")
            print("Epoch: %d, Reward: %.2f" % (epoch, reward))
            logfile.write("%d, %.3f\n" % (epoch, reward))
            logfile.close()
            if reward > best_reward:
                best_reward = reward
                sv.saver.save(sess, os.path.join(args.log, 'best'))
                print("Saved to: %s" % os.path.join(args.log, 'best'))
                
            epoch += 1

    logger.info('tester stopped.')

def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-gpu', '--gpu', default=0, type=int, help='Number of GPUs')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--num-ps', type=int, default=1, help="Number of parameter servers")
    parser.add_argument('--log', default="/tmp/vpn", help='Log directory path')
    parser.add_argument('--env-id', default="maze", help='Environment id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
    parser.add_argument('-a', '--alg', choices=['A3C', 'Q', 'VPN'], default="A3C")
    parser.add_argument('-mo', '--model', type=str, default="LSTM", help="Name of model: [CNN | LSTM]")
    parser.add_argument('--eval-freq', type=int, default=250000, help="Evaluation frequency")
    parser.add_argument('--eval-num', type=int, default=500, help="Evaluation frequency")
    parser.add_argument('--eval-epoch', type=int, default=0, help="Evaluation epoch")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--config', type=str, default="config/collect_deterministic.xml", 
            help="config xml file for environment")

    # Hyperparameters
    parser.add_argument('-n', '--t-max', type=int, default=10, help="Number of unrolling steps")
    parser.add_argument('-g', '--gamma', type=float, default=0.98, help="Discount factor")
    parser.add_argument('-ld', '--ld', type=float, default=1, help="Lambda for GAE")
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--decay', type=float, default=0.95, help="Learning decay")
    parser.add_argument('-ms', '--max-step', type=int, default=int(15e6), help="Max global step")
    parser.add_argument('--dim', type=int, default=0, help="Number of final hidden units")
    parser.add_argument('--f-num', type=str, default='32,32,64', help="num of conv filters")
    parser.add_argument('--f-pad', type=str, default='SAME', help="padding of conv filters")
    parser.add_argument('--f-stride', type=str, default='1,1,2', help="stride of conv filters")
    parser.add_argument('--f-size', type=str, default='3,3,4', help="size of conv filters")
    parser.add_argument('--h-dim', type=str, default='', help="num of hidden units")

    # Q-Learning parameters
    parser.add_argument('-s', '--sync', type=int, default=10000, 
                        help="Target network synchronization frequency")
    parser.add_argument('-f', '--update-freq', type=int, default=1, 
                        help="Parameter update frequency")
    parser.add_argument('--eps-step', type=int, default=int(1e6), 
                    help="Num of local steps for epsilon scheduling")
    parser.add_argument('--eps', type=float, default=0.05, help="Final epsilon value")
    parser.add_argument('--eps-eval', type=float, default=0.0, help="Epsilon for evaluation")

    # VPN parameters
    parser.add_argument('--prediction-step', type=int, default=3, help="number of prediction steps")
    parser.add_argument('--branch', type=str, default="4,4,4", help="branching factor")
    parser.add_argument('--buf', type=int, default=10**6, help="num of steps for random buffer")

    args = parser.parse_args()
    args.f_num = util.parse_to_num(args.f_num)
    args.f_stride = util.parse_to_num(args.f_stride)
    args.f_size = util.parse_to_num(args.f_size)
    args.h_dim = util.parse_to_num(args.h_dim)
    args.eps_eval = min(args.eps, args.eps_eval)
    args.branch = util.parse_to_num(args.branch)
    spec = cluster_spec(args.num_workers, args.num_ps)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    gpu_options = None
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, 
                                            inter_op_parallelism_threads=1,
                                            gpu_options=gpu_options))
        run(args, server)
    elif args.job_name == "test":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, 
                                            inter_op_parallelism_threads=1,
                                            gpu_options=gpu_options))
        run_tester(args, server)
    elif args.job_name == "ps":
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"],
                                 gpu_options=gpu_options))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
