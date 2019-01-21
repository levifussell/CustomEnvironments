'''
This environment will take input from a multi-agent system
(i.e. actions distributed as a list) and convert it to a single-agent
input of a pre-defined environment. The output will then convert
the single-agent output to a multi-agent output
(i.e. states distributed as a list).
'''

import numpy as np
import gym

class MultiAgentToSingleAgentEnv(gym.Env):

    def __init__(self):
        self.__version__ = '0.0.1'

        self.env_single_agent = None

    def load_env(self, env_name, num_agents):
        self.env_single_agent = gym.make(env_name)
        self.num_agents = num_agents

    def step(self, actions):
        assert type(actions) == type([]), "actions for multi-agent environment must be a list"
        assert self.env_single_agent is not None, "an environment must be initialised first with load_env()"

        # convert the list of actions into a single action vector
        action_single_agent = np.stack(actions).tolist()

        next_state, reward, done, info = self.env_single_agent.step(action_single_agent)
        # unstack the state, reward, done
        #next_states = [a.tolist() for a in np.split(next_state, self.num_agents)]
        next_states = [next_state.tolist() for s in range(self.num_agents)]
        rewards = [reward if type(reward) is type(10) else reward[0] for a in range(self.num_agents)]
        dones = [done for a in range(self.num_agents)]

        return next_states, rewards, dones, info

    def reset(self):
        if self.env_single_agent is None:
            return None
        else:
            state = self.env_single_agent.reset()
            states = [state.tolist() for s in range(self.num_agents)]
            return states

    def seed(self, seed=None):
        assert self.env_single_agent is not None, "an environment must be initialised first with load_env()"
        return self.env_single_agent.seed(seed)

    def render(self, mode='human', close=False):
        assert self.env_single_agent is not None, "an environment must be initialised first with load_env()"

        self.env_single_agent.render(mode=mode, close=close)
