# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import nef

from ca.nengo.util import MU


class EnvironmentTemplate(nef.SimpleNode):
    """A template for defining environments to interact with RL agents.

    :input action: vector representing action selected by agent
    :output state: vector representing current state
    :output reward: reward value
    """

    def __init__(self, name, stateD, actions):
        """Initialize environment variables.

        :param name: name for environment
        :param stateD: dimension of state representation
        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        """

        self.actions = actions
        self.state = [0 for _ in range(stateD)]
        self.action = None
        self.reward = 0.0

        nef.SimpleNode.__init__(self, name)
        self.getTermination("action").setDimensions(len(actions[0][1]))

    def termination_action(self, a, pstc=0.01):
        # set the selected action to the one with highest similarity to the
        # current action input
        self.action = max(self.actions, key=lambda x: MU.prod(a, x[1]))

    def origin_state(self):
        return self.state

    def origin_reward(self):
        return [self.reward]

    def tick(self):
        raise NotImplementedError
