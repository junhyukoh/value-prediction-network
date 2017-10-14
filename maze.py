from PIL import Image
import numpy as np
import universe
import gym
import logging
import copy
from bs4 import BeautifulSoup
import tensorflow as tf
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

BLOCK = 0
AGENT = 1
GOAL = 2
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]

COLOR = [[44, 42, 60], # block
        [91, 255, 123], # agent
        [52, 152, 219], # goal
        ]

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def generate_maze(size, holes=0):
    # Source: http://code.activestate.com/recipes/578356-random-maze-generator/
    # Random Maze Generator using Depth-first Search
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm
    mx = size-2; my = size-2 # width and height of the maze
    maze = np.ones((my, mx))
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    # start the maze from a random cell
    start_x = np.random.randint(0, mx); start_y = np.random.randint(0, my)
    cx, cy = 0, 0
    # stack element: (x, y, direction)
    maze[start_y][start_x] = 0; stack = [(start_x, start_y, 0)] 
    while len(stack) > 0:
        (cx, cy, cd) = stack[-1]
        # to prevent zigzags:
        # if changed direction in the last move then cannot change again
        if len(stack) > 2:
            if cd != stack[-2][2]: dirRange = [cd]
            else: dirRange = range(4)
        else: dirRange = range(4)

        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in dirRange:
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 1:
                    ctr = 0 # of occupied neighbors must be 1
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == 0: ctr += 1
                    if ctr == 1: nlst.append(i)

        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[np.random.randint(0, len(nlst))]
            cx += dx[ir]; cy += dy[ir]; maze[cy][cx] = 0
            stack.append((cx, cy, ir))
        else: stack.pop()

    maze_tensor = np.zeros((size, size, 3))
    maze_tensor[:,:,BLOCK] = 1
    maze_tensor[1:-1, 1:-1, BLOCK] = maze
    maze_tensor[start_y+1][start_x+1][AGENT] = 1

    while holes > 0:
        removable = []
        for y in range(0, my):
            for x in range(0, mx):
                if maze_tensor[y+1][x+1][BLOCK] == 1:
                    if maze_tensor[y][x+1][BLOCK] == 1 and maze_tensor[y+2][x+1][BLOCK] == 1 and \
                        maze_tensor[y+1][x][BLOCK] == 0 and maze_tensor[y+1][x+2][BLOCK] == 0:
                            removable.append((y+1, x+1))
                    elif maze_tensor[y][x+1][BLOCK] == 0 and maze_tensor[y+2][x+1][BLOCK] == 0 and \
                        maze_tensor[y+1][x][BLOCK] == 1 and maze_tensor[y+1][x+2][BLOCK] == 1:
                            removable.append((y+1, x+1))
        
        if len(removable) == 0:
            break
        
        idx = np.random.randint(0, len(removable))
        maze_tensor[removable[idx][0]][removable[idx][1]][BLOCK] = 0
        holes -= 1

    return maze_tensor, start_y+1, start_x+1

def find_empty_loc(maze):
    size = maze.shape[0]
    # Randomly determine a goal position
    for i in range(300):
        y = np.random.randint(0, size-2) + 1
        x = np.random.randint(0, size-2) + 1
        if np.sum(maze[y][x]) == 0:
            return [y, x]
    
    raise AttributeError("Cannot find an empty location in 300 trials")

def generate_maze_with_multiple_goal(size, num_goal=1, holes=3):
    maze, start_y, start_x = generate_maze(size, holes=holes)
    
    # Randomly determine agent position
    maze[start_y][start_x][AGENT] = 0
    agent_pos = find_empty_loc(maze)
    maze[agent_pos[0]][agent_pos[1]][AGENT] = 1

    object_pos = [[],[],[],[]]
    for i in range(num_goal):
        pos = find_empty_loc(maze)
        maze[pos[0]][pos[1]][GOAL] = 1
        object_pos[GOAL].append(pos)
    
    return maze, agent_pos, object_pos

def visualize_maze(maze, img_size=320):
    my = maze.shape[0]
    mx = maze.shape[1]
    colors = np.array(COLOR, np.uint8)
    num_channel = maze.shape[2]
    vis_maze = np.matmul(maze, colors[:num_channel])
    vis_maze = vis_maze.astype(np.uint8)
    for i in range(vis_maze.shape[0]):
        for j in range(vis_maze.shape[1]):
            if maze[i][j].sum() == 0.0:
                vis_maze[i][j][:] = int(255)
    image = Image.fromarray(vis_maze) 
    return image.resize((int(float(img_size) * mx / my), img_size), Image.NEAREST)

def visualize_mazes(maze, img_size=320):
    if maze.ndim == 3:
        return visualize_maze(maze, img_size=img_size)
    elif maze.ndim == 4:
        n = maze.shape[0]
        size = maze.shape[1]
        dim = maze.shape[-1]
        concat_m = maze.transpose((1,0,2,3)).reshape((size, n * size, dim))
        return visualize_maze(concat_m, img_size=img_size)
    else:
        raise ValueError("maze should be 3d or 4d tensor")

def to_string(maze):
    my = maze.shape[0]
    mx = maze.shape[1]
    str = ''
    for y in range(my):
        for x in range(mx):
            if maze[y][x][BLOCK] == 1:
                str += '#'
            elif maze[y][x][AGENT] == 1:
                str += 'o'
            elif maze[y][x][GOAL] == 1:
                str += 'x'
            else:
                str += ' '
        str += '\n'
    return str

class Maze(object):
    def __init__(self, size=10, num_goal=1, holes=0):
        self.size = size
        self.dx = [0, 1, 0, -1]
        self.dy = [-1, 0, 1, 0]
        self.num_goal = num_goal
        self.holes = holes
        self.reset()

    def reset(self):
        self.maze, self.agent_pos, self.obj_pos = \
                generate_maze_with_multiple_goal(self.size, num_goal=self.num_goal, 
                        holes=self.holes)
    
    def is_reachable(self, y, x):
        return self.maze[y][x][BLOCK] == 0

    def is_branch(self, y, x):
        if self.maze[y][x][BLOCK] == 1:
            return False
        neighbor_count = 0
        for i in range(4):
            new_y = y + self.dy[i]
            new_x = x + self.dx[i]
            if self.maze[new_y][new_x][BLOCK] == 0:
                neighbor_count += 1
        return neighbor_count > 2
    
    def is_agent_on_branch(self):
        return self.is_branch(self.agent_pos[0], self.agent_pos[1])

    def is_end_of_corridor(self, y, x, direction):
        return self.maze[y + self.dy[direction]][x + self.dx[direction]][BLOCK] == 1
    
    def is_agent_on_end_of_corridor(self, direction):
        return self.is_end_of_corridor(self.agent_pos[0], self.agent_pos[1], direction)

    def move_agent(self, direction):
        y = self.agent_pos[0] + self.dy[direction]
        x = self.agent_pos[1] + self.dx[direction]
        if not self.is_reachable(y, x):
            return False
        self.maze[self.agent_pos[0]][self.agent_pos[1]][AGENT] = 0
        self.maze[y][x][AGENT] = 1
        self.agent_pos = [y, x]
        return True

    def is_object_reached(self, obj_idx):
        if self.maze.shape[2] <= obj_idx:
            return False
        return self.maze[self.agent_pos[0]][self.agent_pos[1]][obj_idx] == 1
    
    def is_empty(self, y, x):
        return np.sum(self.maze[y][x]) == 0

    def remove_object(self, y, x, obj_idx):
        removed = self.maze[y][x][obj_idx] == 1
        self.maze[y][x][obj_idx] = 0
        self.obj_pos[obj_idx].remove([y, x])
        return removed
    
    def remaining_goal(self):
        return self.remaining_object(GOAL)
    
    def remaining_object(self, obj_idx):
        return len(self.obj_pos[obj_idx])

    def add_object(self, y, x, obj_idx):
        if self.is_empty(y, x):
            self.maze[y][x][obj_idx] = 1
            self.obj_pos[obj_idx].append([y, x])
        else:
            ValueError("%d, %d is not empty" % (y, x))
    
    def move_object_random(self, prob, obj_idx):
        pos_copy = copy.deepcopy(self.obj_pos[obj_idx])
        for pos in pos_copy:
            if not hasattr(self, "goal_move_prob"):
                self.goal_move_prob = np.random.rand(1000)
                self.goal_move_idx = 0
            else:
                self.goal_move_idx = (self.goal_move_idx + 1) \
                        % self.goal_move_prob.size
            if self.goal_move_prob[self.goal_move_idx] < prob:
                possible_moves = []
                for i in range(4):
                    y = pos[0] + DY[i]
                    x = pos[1] + DX[i]
                    if self.is_empty(y, x):
                        possible_moves.append(i)
                if len(possible_moves) > 0:
                    self.move_object(pos, obj_idx,
                            possible_moves[np.random.randint(len(possible_moves))])

    def move_object(self, pos, obj_idx, direction):
        y = pos[0] + self.dy[direction]
        x = pos[1] + self.dx[direction]
        if not self.is_reachable(y, x):
            return False
        self.remove_object(pos[0], pos[1], obj_idx)
        self.add_object(y, x, obj_idx)
        return True

    def observation(self, clone=True):
        return np.array(self.maze, copy=clone)

    def visualize(self):
        return visualize_maze(self.maze)
    
    def to_string(self):
        return to_string(self.maze)

class MazeEnv(object):
    def __init__(self, config="", verbose=1):
        self.config = BeautifulSoup(config, "lxml") 
        # map
        self.size = int(self.config.maze["size"])
        self.max_step = int(self.config.maze["time"])
        self.holes = int(self.config.maze["holes"])
        self.num_goal = int(self.config.object["num_goal"])
        # reward
        self.default_reward = float(self.config.reward["default"])
        self.goal_reward = float(self.config.reward["goal"])
        self.lazy_reward = float(self.config.reward["lazy"])
        # randomness
        self.prob_stop = float(self.config.random["p_stop"])
        self.prob_goal_move = float(self.config.random["p_goal"])
        # meta
        self.meta_remaining_time = str2bool(self.config.meta["remaining_time"]) if \
                self.config.meta.has_attr("remaining_time") else False
      
        # log
        self.log_freq = 100
        self.log_t = 0
        self.max_history = 1000
        self.reward_history = []
        self.length_history = []
        self.verbose = verbose

        self.reset()
        self.action_space = gym.spaces.discrete.Discrete(4)
        self.observation_space = gym.spaces.box.Box(0, 1, self.observation().shape)


    def observation(self, clone=True):
        return self.maze.observation(clone=clone)

    def reset(self, reset_episode=True, holes=None):
        if reset_episode:
            self.t = 0
            self.episode_reward = 0
            self.last_step_reward = 0.0
            self.terminated = False

        holes = self.holes if holes is None else holes
        self.maze = Maze(self.size, num_goal=self.num_goal, 
                    holes=holes) 

        return self.observation()

    def remaining_time(self, normalized=True):
        return float(self.max_step - self.t) / float(self.max_step)
    
    def last_reward(self):
        return self.last_step_reward
    
    def meta(self):
        meta = []
        if self.meta_remaining_time:
            meta.append(self.remaining_time())
        if len(meta) == 0:
            return None
        return meta

    def visualize(self):
        return self.maze.visualize() 
    
    def to_string(self):
        return self.maze.to_string()

    def step(self, act):
        assert self.action_space.contains(act), "invalid action: %d" % act
        assert not self.terminated, "episode is terminated"
        self.t += 1

        self.object_reached = False
        self.rand_stopped = False
        if self.prob_stop > 0 and np.random.rand() < self.prob_stop:
            reward = self.default_reward
            self.rand_stopped = True
        else:
            moved = self.maze.move_agent(act)
            reward = self.default_reward if moved else self.lazy_reward

        if self.maze.is_object_reached(GOAL):
            self.object_reached = True
            reward = self.goal_reward
            self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], GOAL)
            if self.maze.remaining_goal() == 0:
                self.terminated = True

        if self.t >= self.max_step:
            self.terminated = True

        self.episode_reward += reward
        self.last_step_reward = reward

        to_log = None
        if self.terminated:
            if self.verbose > 0:
                logger.info('Episode terminating: episode_reward=%s episode_length=%s', 
                            self.episode_reward, self.t)
            self.log_episode(self.episode_reward, self.t)
            if self.log_t < self.log_freq:
                self.log_t += 1
            else:
                to_log = {}
                to_log["global/episode_reward"] = self.reward_mean(self.log_freq)
                to_log["global/episode_length"] = self.length_mean(self.log_freq)
                self.log_t = 0
        else:
            if self.prob_goal_move > 0:
                self.maze.move_object_random(self.prob_goal_move, GOAL) 
                # print("goal_moved")

        return self.observation(), reward, self.terminated, to_log, 1
    
    def log_episode(self, reward, length):
        self.reward_history.insert(0, reward)
        self.length_history.insert(0, length)
        while len(self.reward_history) > self.max_history:
            self.reward_history.pop()
            self.length_history.pop()

    def reward_mean(self, num):
        return np.asarray(self.reward_history[:num]).mean()
    
    def length_mean(self, num):
        return np.asarray(self.length_history[:num]).mean()   
    
    def tf_visualize(self, x):
        colors = np.array(COLOR, np.uint8)
        colors = colors.astype(np.float32) / 255.0
        color = tf.constant(colors)
        obs_dim = self.observation_space.shape[-1]
        x = x[:, :, :, :obs_dim]
        xdim = x.get_shape()
        x = tf.clip_by_value(x, 0, 1)
        bg = tf.ones((tf.shape(x)[0], int(x.shape[1]), int(x.shape[2]), 3))
        w = tf.minimum(tf.expand_dims(tf.reduce_sum(x, axis=xdim.ndims-1), -1), 1.0) 
        w = tf.reshape(tf.tile(w, [1, 1, 1, 3]), tf.shape(bg))
        fg = tf.reshape(tf.matmul(tf.reshape(x, (-1, int(xdim[-1]))), 
                color[:xdim[-1], :]), tf.shape(bg))
        return bg * (1.0 - w) + fg
        
class MazeSMDP(MazeEnv):
    def __init__(self, gamma=0.99, *args, **kwargs):
        super(MazeSMDP, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.prob_slip = float(self.config.random["p_slip"])
    
    def step(self, act):
        assert self.action_space.contains(act), "invalid action: %d" % act
        assert not self.terminated, "episode is terminated"

        reward = 0
        steps = 0
        time = 0
        gamma = 1.0
        self.last_observation = self.maze.observation()
        while not self.terminated:            
            _, r, _, to_log, t = super(MazeSMDP, self).step(act)
            reward += r * gamma
            steps += 1
            time += t
            gamma = gamma * self.gamma
            if not self.rand_stopped:
                if self.maze.is_agent_on_end_of_corridor(act):
                    break
                if self.object_reached:
                    break
                if self.maze.is_agent_on_branch():
                    if self.prob_slip > 0 and np.random.rand() < self.prob_slip:
                        pass
                    else:
                        break

        self.last_step_reward = reward
        return self.observation(), reward, self.terminated, to_log, time
