# Introduction
This repository implements **[NIPS 2017 Value Prediction Network (Oh et al.)](https://arxiv.org/abs/1707.03497)** in Tensorflow.
```
@inproceedings{Oh2017VPN,
  title={Value Prediction Network},
  author={Junhyuk Oh and Satinder Singh and Honglak Lee},
  booktitle={NIPS},
  year={2017}
}
```
Our code is based on [OpenAI's A3C implemenation](https://github.com/openai/universe-starter-agent).

# Dependencies
 * [Tensorflow](https://www.tensorflow.org/install/)
 * [Beutiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
 * [Golang](https://golang.org/doc/install)
 * [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
 * [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
 * [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
 * [gym](https://pypi.python.org/pypi/gym)
 * gym[atari]
 * [universe](https://pypi.python.org/pypi/universe)
 * [opencv-python](https://pypi.python.org/pypi/opencv-python)
 * [numpy](https://pypi.python.org/pypi/numpy)
 * [scipy](https://pypi.python.org/pypi/scipy)

# Training
The following command trains a value prediction network (VPN) with plan depth of 3 on stochastic Collect domain:
```
python train.py --config config/collect_deterministic.xml --branch 4,4,4 --alg VPN
```
`train_vpn` script contains commands for reproducing the main result of the paper.

# Notes
* Tensorboard shows the performance of the epsilon-greedy policy. This is NOT the learning curve in the paper, because epsilon decreases from 1.0 to 0.05 for the first 1e6 steps. Instead, `[logdir]/eval.csv` shows the performance of the agent using greedy-policy.
* Our code supports multi-gpu training. You can specify GPU IDs in `--gpu` option (e.g., `--gpu 0,1,2,3`). 
