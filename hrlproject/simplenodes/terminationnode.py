# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import copy

import nef

from hrlproject.misc import HRLutils
from hrlproject.misc.HRLutils import rand as random


class TerminationNode(nef.SimpleNode):
    """Node used to detect termination of action selected by RL agent, and
    generate appropriate signals to drive agent to respond.

    :input context: current goal that node will decide when to terminate
    :output learn: signal indicating that the agent should learn (if 1)
    :output reset: signal indicating that the agent should reset (if 1)
    :output pseudoreward: pseudoreward value associated with termination
    """

    def __init__(self, conditions, env, contextD=0, name="TerminationNode",
                 rewardval=0.0, state_delay=0.0, learn_interval=0.1,
                 reset_delay=0.1, reset_interval=0.05):
        """Initialize node variables.

        :param conditions: dict mapping contexts to termination states
            associated with that termination
            :type conditions: dict of {Timer:_} and {label:state}
        :param env: environment driving task associated with this node
        :param contextD: dimension of context signal
        :param name: name for the node
        :param rewardval: reward value on successful termination
        :param state_delay: period to wait after termination before starting
            learning/reset
        :param learn_interval: time to learn for
        :param reset_delay: time between learn and reset
        :param reset_interval: time to reset for
        """

        nef.SimpleNode.__init__(self, name)

        self.conds = conditions
        self.rewardval = rewardval
        self.env = env
        self.prev_state = [0]
        self.context = [0]
        # reward value when no termination conditions met
        self.defaultreward = -0.05
        self.reward = self.defaultreward
        self.state_penalty = 0.0
        self.state_delay = state_delay
        self.learn_interval = learn_interval
        self.reset_delay = reset_delay
        self.reset_interval = reset_interval

        # number of timesteps agent has been in termination state
        self.rewardamount = 0
        # number of timesteps agent must spend in termination state before
        # termination is triggered (useful to express it as time in
        # target / dt)
        self.rewardresetamount = 0.5 / 0.001
        self.learntime = [-1, -1]
        # reset right at the beginning to set things up
        self.resettime = [0.05, 0.1]

        def contextf(x, dimensions=contextD, pstc=0.001):
            self.context = copy.deepcopy(x)

        self.create_termination("context", contextf)

        self.create_origin("learn", lambda: [1.0 if self.t > self.learntime[0]
                                             and self.t < self.learntime[1]
                                             else 0.0])
        self.create_origin("reset", lambda: [1.0 if self.t > self.resettime[0]
                                             and self.t < self.resettime[1]
                                             else 0.0])
        self.create_origin("pseudoreward", lambda: [self.reward])

    def tick(self):
        cond_active = False
        for c in self.conds:
            if isinstance(c, Timer):
                # if it is a timer entry, just update the timer and check if it
                # has expired
                c.tick()
                if c.ring():
                    self.reward = self.rewardval
                    self.activate()
                    c.reset()
                    cond_active = True

            elif (self.env.is_in(self.env.state, c) and
                  (self.conds[c] is None or
                   HRLutils.similarity(HRLutils.normalize(self.context),
                                       self.conds[c]) > 0.3)):
                # if it is a state entry, check if the agent is in the region
                # associated with that state, and check if that region is the
                # one corresponding to the currently selected context

                self.reward = self.rewardval

                self.rewardamount += 1
                if self.rewardamount > self.rewardresetamount:
                    self.activate()
                    self.rewardamount = 0

                cond_active = True

        # if no termination conditions met, just give default reward
        if not cond_active:
            self.reward = self.defaultreward

        # reset rewardamount when the reset signal is sent (so that there won't
        # be any leftover rewardamount from the agent's previous decision)
        if self.t > self.resettime[0] and self.t < self.resettime[1]:
            self.rewardamount = 0

        # add a penalty if the state hasn't changed (to help prevent agent from
        # getting stuck)
        if sum(self.prev_state) != 0 and \
                HRLutils.similarity(HRLutils.normalize(self.env.state),
                                    HRLutils.normalize(self.prev_state)) < 1.0:
            self.state_penalty = 0.0
        else:
            self.state_penalty += 0.0001
        self.prev_state = copy.deepcopy(self.env.state)

        self.reward = self.reward - self.state_penalty

    def activate(self):
        self.learntime = [self.t + self.state_delay,
                          self.t + self.state_delay + self.learn_interval]
        self.resettime = [self.learntime[1] + self.reset_delay,
                          self.learntime[1] + self.reset_delay +
                          self.reset_interval]


class Timer:
    """A simple timer that counts down from some point with a given dt."""

    def __init__(self, period, dt=0.001):
        self.period = period
        self.dt = dt
        self.time = 0

        self.reset()

    def reset(self):
        self.time = random.uniform(self.period[0], self.period[1])

    def tick(self):
        self.time -= self.dt

    def ring(self):
        return self.time <= 0.0
