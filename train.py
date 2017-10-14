import argparse
import os
import sys
from six.moves import shlex_quote
import util

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-gpu', '--gpu', default="", type=str, help='GPU Ids')
parser.add_argument('-w', '--num-workers', type=int, default=16, help="Number of workers")
parser.add_argument('-ps', '--num-ps', type=int, default=4, help="Number of parameter servers")
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and rewarders to use'
                    '(e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="maze",
                    help="Environment id")
parser.add_argument('-l', '--log', type=str, default="result/maze", 
                    help="Log directory path")
parser.add_argument('-d', '--dry-run', action='store_true',
                    help="Print out commands rather than executing them")
parser.add_argument('-m', '--mode', type=str, default='tmux',
                    help="tmux: run workers in a tmux session. "
                    "nohup: run workers with nohup. "
                    "child: run workers as child processes")
parser.add_argument('-a', '--alg', choices=['A3C', 'Q', 'VPN'], default="VPN")
parser.add_argument('-mo', '--model', type=str, default="CNN", help="Name of model: [CNN | LSTM]")
parser.add_argument('-ms', '--max-step', type=int, default=int(15e6), help="Max global step")
parser.add_argument('--config', type=str, default="config/collect_deterministic.xml", 
        help="config xml file for environment")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--eval-freq', type=int, default=250000, help="Evaluation frequency")
parser.add_argument('--eval-num', type=int, default=2000, help="Evaluation frequency")

# Hyperparameters
parser.add_argument('-n', '--t-max', type=int, default=10, help="Number of unrolling steps")
parser.add_argument('-g', '--gamma', type=float, default=0.98, help="Discount factor")
parser.add_argument('-lr', '--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--decay', type=float, default=0.95, help="Learning rate")
parser.add_argument('--dim', type=int, default=64, help="Number of final hidden units")
parser.add_argument('--f-num', type=str, default='32,32,64', help="num of conv filters")
parser.add_argument('--f-stride', type=str, default='1,1,2', help="stride of conv filters")
parser.add_argument('--f-size', type=str, default='3,3,4', help="size of conv filters")
parser.add_argument('--f-pad', type=str, default='SAME', help="padding of conv filters")
parser.add_argument('--h-dim', type=str, default='', help="num of hidden units")

# Q-Learning parameters
parser.add_argument('-s', '--sync', type=int, default=10000, 
                    help="Target network synchronization frequency")
parser.add_argument('-f', '--update-freq', type=int, default=1, 
                    help="Parameter update frequency")
parser.add_argument('--eps-step', type=int, default=int(1e6), 
                    help="Num of local steps for epsilon scheduling")
parser.add_argument('--eps', type=float, default=0.05, help="Final epsilon")
parser.add_argument('--eps-eval', type=float, default=0.0, help="Epsilon for evaluation")

# VPN parameters
parser.add_argument('--prediction-step', type=int, default=3, help="number of prediction steps")
parser.add_argument('--branch', type=str, default="4,4,4", help="branching factor")
parser.add_argument('--buf', type=int, default=10**6, help="num of steps for random buffer")

def new_cmd(session, name, cmd, mode, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    if mode == 'tmux':
        return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))
    elif mode == 'child':
        return name, "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, logdir, session, name, logdir)
    elif mode == 'nohup':
        return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd), logdir, session, name, logdir)


def create_commands(session, args, shell='bash'):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        sys.executable, 'worker.py',
        '--log', args.log, '--env-id', args.env_id,
        '--num-workers', str(args.num_workers),
        '--num-ps', str(args.num_ps),
        '--alg', args.alg,
        '--model', args.model,
        '--max-step', args.max_step,
        '--t-max', args.t_max,
        '--eps-step', args.eps_step,
        '--eps', args.eps,
        '--eps-eval', args.eps_eval,
        '--gamma', args.gamma,
        '--lr', args.lr,
        '--decay', args.decay,
        '--sync', args.sync,
        '--update-freq', args.update_freq,
        '--eval-freq', args.eval_freq,
        '--eval-num', args.eval_num,
        '--prediction-step', args.prediction_step,
        '--dim', args.dim,
        '--f-num', args.f_num,
        '--f-pad', args.f_pad,
        '--f-stride', args.f_stride,
        '--f-size', args.f_size,
        '--h-dim', args.h_dim,
        '--branch', args.branch,
        '--config', args.config,
        '--buf', args.buf,
        ]

    if len(args.gpu) > 0:
        base_cmd += ['--gpu', 1]

    if args.remotes is None:
        args.remotes = ["1"] * args.num_workers
    else:
        args.remotes = args.remotes.split(',')
        assert len(args.remotes) == args.num_workers
    
    cmds_map = []
    for i in range(args.num_ps):
        prefix = ['CUDA_VISIBLE_DEVICES=']
        cmds_map += [new_cmd(session, "ps-%d" % i, prefix + base_cmd + ["--job-name", "ps", 
                    "--task", str(i)], args.mode, args.log, shell)]

    for i in range(args.num_workers):
        prefix = []
        if len(args.gpu) > 0:
            prefix = ['CUDA_VISIBLE_DEVICES=%d' % args.gpu[(i % len(args.gpu))]]
        else:
            prefix = ['CUDA_VISIBLE_DEVICES=']
        cmds_map += [new_cmd(session,
                "w-%d" % i, 
                prefix + base_cmd + ["--job-name", "worker", "--task", str(i), 
                "--remotes", args.remotes[i]], args.mode, args.log, shell)]
    
    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", args.log, 
          "--port", "12345"], args.mode, args.log, shell)]
    cmds_map += [new_cmd(session, "test", prefix + base_cmd + ["--job-name", "test", 
                "--task", str(args.num_workers)], args.mode, args.log, shell)]
    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(args.log),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), args.log),
    ]
    if args.mode == 'nohup' or args.mode == 'child':
        cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(args.log)]
        notes += ["Run `source {}/kill.sh` to kill the job".format(args.log)]
    if args.mode == 'tmux':
        notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
        notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    else:
        notes += ["Use `tail -f {}/*.out` to watch process output".format(args.log)]
    notes += ["Point your browser to http://localhost:12345 to see Tensorboard"]

    if args.mode == 'tmux':
        cmds += [
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
        ]
        for w in windows[1:]:
            cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
        cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds, notes


def run():
    args = parser.parse_args()
    args.gpu = util.parse_to_num(args.gpu)
    cmds, notes = create_commands("e", args)
    if args.dry_run:
        print("Dry-run mode due to -d flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")
    print("\n".join(cmds))
    print("")
    if not args.dry_run:
        if args.mode == "tmux":
            os.environ["TMUX"] = ""
        path = os.path.join(os.getcwd(), args.log)
        if os.path.exists(path):
            key = raw_input("%s exists. Do you want to delete it? (y/n): " % path)
            if key != 'n':
                os.system("rm -rf %s" % path)
                os.system("\n".join(cmds))
                print('\n'.join(notes))
        else:
            os.system("\n".join(cmds))
            print('\n'.join(notes))


if __name__ == "__main__":
    run()
