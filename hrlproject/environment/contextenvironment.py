# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import math

from ca.nengo.util import MU

from hrlproject.environment.placecell_bmp import PlaceCellEnvironment
from hrlproject.misc import HRLutils
from hrlproject.misc.HRLutils import rand as random


class ContextEnvironment(PlaceCellEnvironment):
    """Environment based on PlaceCellEnvironment, but supplemented with a
    context signal.

    :input action: vector representing action selected by agent
    :output state: current x,y location of agent
    :output reward: reward value
    :output place: vector concatenating activation of each place cell
    :output optimal_move: the ideal action for the agent to take in the current
        state
    :output context: vector representing current context
    :output placewcontext: concatenation of place and context
    """

    def __init__(self, actions, mapname, contextD, context_rewards, **kwargs):
        """Initialize the environment variables.

        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param mapname: filename for map file
        :param contextD: dimension of vector representing context
        :param context_rewards: mapping from region labels to rewards for being
            in that region (each entry represents one context)
            :type context_rewards: dict {"regionlabel":rewardval,...}
        :param **kwargs: see PlaceCellEnvironment.__init__
        """

        PlaceCellEnvironment.__init__(self, actions, mapname,
                                      name="ContextEnvironment", **kwargs)

        self.rewards = context_rewards

        # generate vectors representing each context
        self.contexts = {}  # mapping from region label to context vector
        for i, r in enumerate(self.rewards):
            self.contexts[r] = list(MU.I(contextD)[i])

        self.context = self.contexts[random.choice(self.contexts.keys())]

        # randomly pick a new context every context_delay seconds
        self.context_delay = 60
        self.context_update = self.context_delay

        self.create_origin("placewcontext",
                           lambda: self.place_activations + self.context)
        self.create_origin("context", lambda: self.context)

    def tick(self):
        PlaceCellEnvironment.tick(self)

        self.update_context()

    def update_reward(self):
        # agent is rewarded if it is in the target region associated with the
        # current context
        self.reward = self.defaultreward
        for r in self.rewards:
            if self.context == self.contexts[r] and self.is_in(self.state, r):
                self.reward += self.rewards[r]

        # penalize for trying to move into walls
        if self.is_in(self.dest, "wall"):
            self.reward = -0.1

        self.rewardamount += 1 if self.reward > 0 else 0

    def update_context(self):
        if self.t > self.context_update:
            self.context = self.contexts[random.choice(self.contexts.keys())]
            self.context_update = self.t + self.context_delay

    def gen_encoders(self, N, contextD, context_scale):
        """Generates encoders for state population in RL agent.

        State aspect of encoders comes from PlaceCellEnvironment. Context
        component is a unit vector with contextD dimensions and length
        context_scale.
        """

        s_encoders = PlaceCellEnvironment.gen_encoders(self, N)
        c_encoders = [random.choice(MU.I(contextD)) for _ in range(N)]
        c_encoders = [[x * context_scale for x in enc] for enc in c_encoders]
        encoders = [s + list(c) for s, c in zip(s_encoders, c_encoders)]
        encoders = [[x / math.sqrt(sum([y ** 2 for y in e])) for x in e]
                    for e in encoders]
        return encoders

    def calc_optimal_move(self):
        """Calculate the optimal move for the agent to take in the current
        state/context."""

        # basically the same as PlaceCellEnvironment.calc_optimal_move, except
        # we look at the current context to find the goal

        goal = [c for c in self.contexts
                if self.contexts[c] == self.context][0]

        stepsize = 0.1
        self.optimal_move = None
        for y in [v * stepsize for v in range(int(-self.imgsize[1] /
                                                  (2 * stepsize)) + 1,
                                              int(self.imgsize[1] /
                                                  (2 * stepsize)) - 1)]:
            for x in [v * stepsize for v in range(int(-self.imgsize[0] /
                                                      (2 * stepsize)) + 1,
                                                  int(self.imgsize[0] /
                                                      (2 * stepsize)) - 1)]:
                if self.is_in((x, y), goal):
                    angle = math.atan2(y - self.state[1], x - self.state[0])
                    pt = (math.cos(angle), math.sin(angle))
                    self.optimal_move = max(
                        self.actions, key=lambda x:-1 if
                        self.is_in((x[1][0] * self.dx + self.state[0],
                                    x[1][1] * self.dx + self.state[1]),
                                   "wall")
                        else HRLutils.similarity(x[1], pt))[0]
                    return

    def colour_translation(self, c):
        """Translate box labels into colours (used for interactivemode
        display)."""

        if c == "a":
            return "yellow"
        if c == "b":
            return "red"
        if c == "wall":
            return "black"
        if c == "floor":
            return "white"
        else:
            return c
