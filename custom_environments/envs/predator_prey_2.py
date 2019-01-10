import gym
import custom_environments
from gym.utils import seeding

from copy import deepcopy
import time
import numpy as np

import curses

class PredatorPrey2(gym.Env):

    def __init__(self):
        self.__version__ = '0.0.1'
        self.seed()

        # reward constants
        self.TIME_PENALTY = -0.5
        self.PREY_REWARD = +10.0 - self.TIME_PENALTY

        # grid information
        self.GRID_DIM = 10

        # predator parameters
        self.num_predators = 2
        self.predator_positions = [[-1, -1]]*self.num_predators

        # prey parameters
        # (TODO: multiply prey targets)
        self.prey_position = [-1, -1]

        # action definition
        self.AGENT_ACTIONS = {
                            0 : [0, -1], # up
                            1 : [1, 0], #right
                            2 : [0, 1], #down
                            3 : [-1, 0], #left
                            4 : [0, 0] #stay
                        }

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)

    def __diff_pos__(self, pos_target, pos_origin):
        '''
        Gets the vector between an origin and target position
        '''
        assert len(pos_target) == len(pos_origin), "two positions must have the same length. target: {}, origin: {}".format(pos_target, pos_origin)
        diff = [a - b for a,b in zip(pos_target,pos_origin)]
        return diff

    def __add_pos__(self, pos_a, pos_b):
        '''
        Adds two positions.
        '''
        assert len(pos_a) == len(pos_b), "two positions must have the same length. a: {}, b: {}".format(pos_a, pos_b)
        addi = [a + b for a,b in zip(pos_a,pos_b)]
        return addi

    def __bound_pos__(self, pos):
        '''
        Stops the agent from moving beyond the edge of the screen.
        '''
        pos[0] = min(max(0, pos[0]), self.GRID_DIM-1)
        pos[1] = min(max(0, pos[1]), self.GRID_DIM-1)
        return pos

    def __get_obs__(self):
        #TODO: for now all predators have full vision and can
        #  see each other clearly.

        # add each predator's own position
        obs_all_predators = [p[:] for p in self.predator_positions]

        # add each other predator's position
        for a in range(self.num_predators):
            for p in range(self.num_predators):
                if a != p:
                    obs_all_predators[a].extend(self.__diff_pos__(
                                                self.predator_positions[p],
                                                self.predator_positions[a]
                                                ))

            # add the position of the prey
            obs_all_predators[a].extend(self.__diff_pos__(
                                        self.prey_position,
                                        self.predator_positions[a]
                                        ))

        assert len(obs_all_predators) == self.num_predators, "observation information incorrect: {}".format(obs_all_predators)

        # normalise the obs
        obs_all_norm = [(np.array(o).astype(float)/float(self.GRID_DIM)).tolist() for o in obs_all_predators]

        return obs_all_norm

    def __act__(self, actions):
        '''
        Performs the actions of the predators.
        '''
        assert len(actions) == self.num_predators, "number of actions must be equal to number of predators: {}".format(actions)

        #TODO: for now, actions are forced to be discrete. 
        for a in range(len(actions)):
            pos_to_add = self.AGENT_ACTIONS[np.argmax(actions[a])]
            self.predator_positions[a] = self.__bound_pos__(
                                        self.__add_pos__(self.predator_positions[a], pos_to_add)
                                        )

    def __get_reward_and_done__(self):
        '''
        Reward is -T for every timestep and +G for spending a timestep at
        the prey.

        reward  : list of rewards for each predator.
        dones   : list of terminations of the env. For now it is the same for all predators.
        '''

        single_rewards = [
                self.TIME_PENALTY + \
                self.PREY_REWARD * np.all(np.equal(p, self.prey_position)).astype(float) \
                for p in self.predator_positions
                ]

        total_reward = np.sum(single_rewards)

        # the reward is the sum of all predator rewards so that
        #  cooperation is encouraged.
        #  TODO: test behaviours with and without coop. rewards.
        coop_rewards = [total_reward]*self.num_predators

        # if every predator reaches the reward the environment terminates
        dones = [total_reward == (self.TIME_PENALTY + self.PREY_REWARD)*self.num_predators]*self.num_predators

        assert len(coop_rewards) == self.num_predators, "reward information incorrect: {}".format(rewards)

        return coop_rewards, dones

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #for p in range(len(self.predator_positions)):
        #    self.predator_positions[p][0] = self.np_random.randint(self.GRID_DIM)
        #    self.predator_positions[p][1] = self.np_random.randint(self.GRID_DIM)
        self.predator_positions = self.np_random.randint(self.GRID_DIM, size=(self.num_predators, 2)).tolist()

        #self.prey_position[0] = self.np_random.randint(self.GRID_DIM)
        #self.prey_position[1] = self.np_random.randint(self.GRID_DIM)
        self.prey_position = self.np_random.randint(self.GRID_DIM, size= 2).tolist()

        obs_all = self.__get_obs__()
        return obs_all

    def step(self, actions):
        '''
        actions     : list of actions for each agent.

        states      : list of states for each agent.
        rewards     : list of rewards for each agent.
        dones       : list of terminals for each agent
                    (this is the same for each agent).
        info        : info. from the environment.
        '''
        
        # perform actions
        self.__act__(actions)

        # get the reward
        rewards, dones = self.__get_reward_and_done__()

        # get the next state/observation
        obs = self.__get_obs__()

        #TODO: information about the step.
        info = None

        return obs, rewards, dones, info

    def render(self, mode='human', close=False):
        '''
        adapted from: https://github.com/IC3Net/IC3Net/blob/master/ic3net-envs/ic3net_envs/predator_prey_env.py
        '''
        grid = np.zeros((self.GRID_DIM, self.GRID_DIM), dtype=object)
        self.stdscr.clear()

        PRED_ICON = 'X'
        PREY_ICON = 'O'
        SPACE_ICON = '-'
        
        # draw predator
        for p in self.predator_positions:
            if grid[p[0],p[1]] != 0:
                grid[p[0],p[1]] = str(grid[p[0],p[1]]) + PRED_ICON
            else:
                grid[p[0],p[1]] = PRED_ICON

        # draw prey
        if grid[self.prey_position[0],self.prey_position[1]] != 0:
            grid[self.prey_position[0],self.prey_position[1]] = \
                    str(grid[self.prey_position[0],self.prey_position[1]]) + PREY_ICON
        else:
            grid[self.prey_position[0],self.prey_position[1]] = PREY_ICON

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if PRED_ICON in item and PREY_ICON in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif PRED_ICON in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, SPACE_ICON.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

if __name__ == '__main__':
    env = gym.make('predator-prey-v2')
    env.init_curses()
    _ = env.reset()
    for i in range(1000):
        acts = [np.random.rand(5).tolist() for _ in range(2)] 

        env.render()
        obvs, rewards, dones, _ = env.step(acts)

        time.sleep(0.3)
        #print(obvs, rewards, dones)
