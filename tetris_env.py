import numpy as np
from scipy.signal import convolve2d
import gym
from gym import spaces
from gym.utils import seeding

import bricks

# Characters for terminal rendering
CHARMAP = {
    -1: '\u25a1 ', # Potential brick
    0: '. ', # Empty space
    1: '\u25a0 ', # Occupied space
}

matrix_to_chars = np.vectorize(lambda x: CHARMAP[x])


class ExplicitTuple(gym.Space):
    """
    (s1, s2, ..., sn) 
    Example usage:
    self.observation_space = ExplicitTuple((1, 3, 5, 7))
    """

    def __init__(self, state_tuple):
        self.state_tuple = state_tuple
        self.n = len(self.state_tuple)
        gym.Space.__init__(self, (self.n,), None)

    def sample(self):
        idx = gym.spaces.np_random.randint(self.n)
        return self.state_tuple[idx]

    def contains(self, x):
        return x in self.state_tuple

    def __getitem__(self, idx):
        return self.state_tuple[idx]

    def __repr__(self):
        return "ExplicitTuple%s" % self.state_tuple.__repr__()

    def __eq__(self, other):
        return self.state_tuple == other.n


class NoStateException(Exception):
    pass


class ActionNotAllowedException(Exception):
    pass


class TetrisEnv(gym.Env):
    """Simple tetris environment
    This environment is limited to finding the optimal placement of the brick
    The required action is the number of rotations + the number of right moves
    from upper right corner of the game window.
    Each state consists of a binary 2d matrix (the game window) and the next brick
    (integer, index in the list of bricks available in the environment)
    """
    metadata = {'render.modes': ['human', 'ansi']}
    EPISODE_END_REWARD = 0
    LINE_REWARD = 1
    BRICK_PLACED_REWARD = 0
    
    def __init__(self, shape=(22,10), possible_bricks=bricks.BRICKS):
        self.shape = shape
        self.height = shape[0]
        self.width = shape[1]
        
        self.bricks = possible_bricks
        self.n_bricks = len(self.bricks)
        self.brick_probabilities = [1./self.n_bricks for b in self.bricks]
        self.brick_height = [b.shape.shape[0] for b in self.bricks]
        self.brick_vpad = [max(self.brick_height) - h for h in self.brick_height]
        self.max_brick_dim = max(max(b.shape.shape) for b in self.bricks)
        
        self.board_space = spaces.MultiBinary(self.shape)
        self.brick_space = spaces.Discrete(self.n_bricks)
        
        self.observation_space = spaces.Tuple((
            self.board_space,
            self.brick_space))
        self.dtype = self.board_space.sample().dtype
        self.seed()

        # Start the first game
        self.reset()
        self._action_space = None
    
    @property
    def action_space(self):
        if self._action_space is None:
            actions = []
            for r, f in enumerate(self.bricks[self.next_brick].filters):
                conv = convolve2d(self.board, f, mode='valid')
                conv = np.vstack([conv, np.ones(conv.shape[1], dtype=self.dtype)])
                vertical_placement = np.argmax(conv > 0, axis=0) - 1
                for x, y in enumerate(vertical_placement):
                    if y == -1:
                        continue
                    actions.append((r, x, y))
            if len(actions) == 0:
                raise NoStateException('No possible actions')
            self._action_space = ExplicitTuple(tuple(actions))
        return self._action_space
    
    @property
    def nA(self):
        try:
            return self.action_space.n
        except NoStateException:
            return 0
    
    @property
    def is_done(self):
        return self.board[0,:].sum() > 0
    
    @property
    def hashable_state(self):
        return (self.board.tostring(), self.next_brick)
    
    def score_and_update(self):
        full_rows = np.sum(self.board, axis=1) == self.shape[1]
        score = full_rows.sum()
        
        self.board[score:, :] = self.board[~full_rows.flatten(), :]
        self.board[:score, :] = 0
        self.episode_score += score
        
        done = False
        if self.is_done:
            score = TetrisEnv.EPISODE_END_REWARD
            done = True
        elif score == 0:
            score = TetrisEnv.BRICK_PLACED_REWARD
        else:
            score *= TetrisEnv.LINE_REWARD
        
        return score, done
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, score=True, return_hashable_state=False):
        #assert self.action_space.contains(action)
        if action is not None:
            r, x, y = action
            brick = self.bricks[self.next_brick].rotations[r]
            assert (self.board[y:y+brick.shape[0], x:x+brick.shape[1]] * brick).sum() == 0
            self.board[y:y+brick.shape[0], x:x+brick.shape[1]] += brick[:]
            self._action_space = None
        self.next_brick = self.np_random.randint(self.n_bricks)
        
        if score:
            reward, done = self.score_and_update()
        else:
            reward = 0
            done = self.is_done
        return self._get_obs(return_hashable_state), reward, done, {}

    def _get_obs(self, hashable=False):
        if hashable:
            return (self.board.tostring(), self.next_brick)
        else:
            return (self.board, self.next_brick)

    def reset(self, return_hashable_state=False):
        self.board = np.zeros(self.shape, dtype=int)
        self.next_brick = self.brick_space.sample()
        self.episode_score = 0
        return self._get_obs(return_hashable_state)

    def render(self, mode='human', action=None):
        if mode == 'ansi':
            if action is None or action[2] < 0:
                board = self.board
            else:
                r, x, y = action
                brick = self.bricks[self.next_brick].rotations[r]
                board = self.board.copy()
                board[y:y+brick.shape[0], x:x+brick.shape[1]] -= brick[:]
            brick = self.bricks[self.next_brick].shape
            lines = ['' for _ in range(self.brick_vpad[self.next_brick])]
            lines.extend(''.join(line) for line in matrix_to_chars(brick))
            lines.append('')
            lines.extend(
                '|' + ''.join(line) + '|' for line in matrix_to_chars(board))
            return '\n'.join(lines)
        elif mode is 'human':
            print(self.render(mode='ansi', action=action))
        else:
            super(TetrisEnv, self).render(mode=mode)
    
    def render_episode(self, policy, hashable_state=False, action_as_idx=True, delay_fn=None):
        next_state = self.reset(return_hashable_state=hashable_state)
        done = False
        while not done:
            try:
                action = policy(next_state)
                if action_as_idx:
                    action = self.action_space[action]
            except NoStateException:
                break
            self.render(action=action)
            next_state, reward, done, _ = self.step(action, return_hashable_state=hashable_state)
            if delay_fn is not None:
                delay_fn()


class SimpleTetrisEnv(TetrisEnv):

    def __init__(self, shape=(22,10), possible_bricks=bricks.BRICKS):
        super(SimpleTetrisEnv, self).__init__(shape=shape, possible_bricks=possible_bricks)
        self._action_space = spaces.MultiDiscrete((4, self.width))
        
    @property
    def nA(self):
        return np.product(self._action_space.nvec)
    
    @property
    def is_done(self):
        """Done is determined on brick placement action
        """
        return self._done
    
    
    def vertical_brick_pos(self, action):
        #assert action in self.action_space
        r, x = action
        try:
            f = self.bricks[self.next_brick].filters[r]
            col = self.board[:, x: x+f.shape[1]]
        except IndexError:
            raise ActionNotAllowedException()
        #print('\n'.join(''.join(line) for line in matrix_to_chars(col)))
        #print()
        #print('\n'.join(''.join(line) for line in matrix_to_chars(f)))
        #print(action)
        conv = convolve2d(col, f, mode='valid')
        conv = np.vstack([conv, np.ones(conv.shape[1], dtype=self.dtype)])
        return np.argmax(conv.flatten() > 0) - 1
    
    def step(self, action, score=True, return_hashable_state=False):
        if not hasattr(action, '__len__'):
            action = (int(action/4), action % 4)
        try:
            y = self.vertical_brick_pos(action)
        except ActionNotAllowedException:
            return self._get_obs(return_hashable_state), 0, self.is_done, {}
        if y == -1:
            self._done = True
            action = None
        else:
            r, x = action
            action = (r, x, y)
        
        return super(SimpleTetrisEnv, self).step(
            action,
            score=score,
            return_hashable_state=return_hashable_state)
    
    def reset(self, return_hashable_state=False):
        self._done = False
        return super(SimpleTetrisEnv, self).reset(
            return_hashable_state=return_hashable_state)
    
    def render(self, mode='human', action=None):
        if len(action) == 2:
            y = self.vertical_brick_pos(action)
            r, x = action
            action = (r, x, y)
            
        return super(SimpleTetrisEnv, self).render(
            mode=mode, action=action)
    
    def random_action(self, r=None):
        if r is None:
            r = self.np_random.randint(len(self.bricks[self.next_brick].rotations))
        width = self.bricks[self.next_brick].rotations[r].shape[1]
        x = env.np_random.randint(self.board.shape[1] + 1 - width)
        return r, x

if __name__ == '__main__':
    #e = TetrisEnv()
    mode = 'simple'
    #mode = 'regular'
    
    if mode == 'regular':
        env = TetrisEnv((8,4), [bricks.T])
        env.seed(2)
        def random_policy(state):
            try:
                return env.np_random.randint(env.nA)
            except NoStateException:
                return 0
        action_as_idx = False
    elif mode == 'simple':
        env = SimpleTetrisEnv((8,4), [bricks.T])
        env.seed(2)
        def random_policy(state):
            return env.random_action()
        action_as_idx = False
    env.render_episode(random_policy, delay_fn=input, action_as_idx=action_as_idx)
