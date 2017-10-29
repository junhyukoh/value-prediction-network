import argparse
import go_vncdriver
import tensorflow as tf
from envs import create_env
import subprocess as sp
import util
import model
import numpy as np
from worker import new_env

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-gpu', '--gpu', default=0, type=int, help='Number of GPUs')
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and rewarders to use'
                    '(e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="maze",
                    help="Environment id")
parser.add_argument('-a', '--alg', type=str, default="VPN", help="Algorithm: [A3C | Q | VPN]")
parser.add_argument('-mo', '--model', type=str, default="CNN", help="Name of model: [CNN | LSTM]")
parser.add_argument('-ck', '--checkpoint', type=str, default="", help="Path of the checkpoint")
parser.add_argument('-n', '--n-play', type=int, default=1000, help="Num of play")
parser.add_argument('--eps', type=float, default=0.0, help="Epsilon-greedy")
parser.add_argument('--config', type=str, default="", help="config xml file for environment")
parser.add_argument('--seed', type=int, default=0, help="Random seed")

# Hyperparameters
parser.add_argument('-g', '--gamma', type=float, default=0.98, help="Discount factor")
parser.add_argument('--dim', type=int, default=64, help="Number of final hidden units")
parser.add_argument('--f-num', type=str, default='32,32,64', help="num of conv filters")
parser.add_argument('--f-stride', type=str, default='1,1,2', help="stride of conv filters")
parser.add_argument('--f-size', type=str, default='3,3,4', help="size of conv filters")
parser.add_argument('--f-pad', type=str, default='SAME', help="padding of conv filters")

# VPN parameters
parser.add_argument('--branch', type=str, default="4,4,4", help="branching factor")

def evaluate(env, agent, num_play=3000, eps=0.0):
    env.max_history = num_play
    for iter in range(0, num_play):
        last_state = env.reset()
        last_features = agent.get_initial_features()
        last_meta = env.meta()
        while True:
            # import pdb; pdb.set_trace()
            if eps == 0.0 or np.random.rand() > eps:
                fetched = agent.act(last_state, last_features,
                        meta=last_meta)
                if agent.type == 'policy':
                    action, features = fetched[0], fetched[2:]
                else:
                    action, features = fetched[0], fetched[1:]
            else:
                act_idx = np.random.randint(0, env.action_space.n)
                action = np.zeros(env.action_space.n)
                action[act_idx] = 1
                features = []

            state, reward, terminal, info, _ = env.step(action.argmax())
            last_state = state
            last_features = features
            last_meta = env.meta()
            if terminal:
                break

    return env.reward_mean(num_play)

def run():
    args = parser.parse_args()
    args.task = 0
    args.f_num = util.parse_to_num(args.f_num)
    args.f_stride = util.parse_to_num(args.f_stride)
    args.f_size = util.parse_to_num(args.f_size)
    args.branch = util.parse_to_num(args.branch)

    env = new_env(args)
    args.meta_dim = 0 if env.meta() is None else len(env.meta())
    device = '/gpu:0' if args.gpu > 0 else '/cpu:0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(device_filters=device, 
            gpu_options=gpu_options,
            allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        if args.alg == 'A3C': 
            model_type = 'policy'
        elif args.alg == 'Q':
            model_type = 'q'
        elif args.alg == 'VPN':
            model_type = 'vpn'
        else:
            raise ValueError('Invalid algorithm: ' + args.alg)
        with tf.device(device):
            with tf.variable_scope("local/learner"):
                agent = eval("model." + args.model)(env.observation_space.shape, 
                    env.action_space.n, type=model_type, 
                    gamma=args.gamma, 
                    dim=args.dim,
                    f_num=args.f_num,
                    f_stride=args.f_stride,
                    f_size=args.f_size,
                    f_pad=args.f_pad,
                    branch=args.branch,
                    meta_dim=args.meta_dim)
                print("Num parameters: %d" % agent.num_param)
        
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)
        np.random.seed(args.seed)
        reward = evaluate(env, agent, args.n_play, eps=args.eps)
        print("Reward: %.2f" % (reward))

if __name__ == "__main__":
    run()
