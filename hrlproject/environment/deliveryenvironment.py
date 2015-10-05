# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import math

from java.awt import Color
from java.awt.image import BufferedImage

from ca.nengo.util import MU

from hrlproject.environment.placecell_bmp import PlaceCellEnvironment
from hrlproject.misc import HRLutils
from hrlproject.misc.HRLutils import rand as random


class DeliveryEnvironment(PlaceCellEnvironment):
    """Environment for the delivery task, where the agent must move to one
    location to pick up a 'package' and another location to drop it off.

    :input action: vector representing action selected by agent
    :output state: current x,y location of agent
    :output reward: reward value
    :output place: vector concatenating activation of each place cell
    :output optimal_move: the ideal action for the agent to take in the current
        state
    :output context: vector representing whether the agent has the package in
        hand
    :output placewcontext: concatenation of place and context
    """

    def __init__(self, *args, **kwargs):
        """Initialize environment variables.

        :param name: name for environment
        :param *args: see PlaceCellEnvironment.__init__
        :param **kwargs: see PlaceCellEnvironment.__init__
        """

        PlaceCellEnvironment.__init__(self, name="DeliveryEnvironment", *args,
                                      **kwargs)

        # reward value when no reward condition is met
        self.defaultreward = -0.05

        self.contexts = {"in_hand": [1, 0], "out_hand": [0, 1]}
        self.in_hand = False

        self.create_origin("placewcontext",
                           lambda: (self.place_activations +
                                    self.contexts["in_hand"] if self.in_hand
                                    else self.place_activations +
                                    self.contexts["out_hand"]))
        self.create_origin("context", lambda: (self.contexts["in_hand"]
                                               if self.in_hand else
                                               self.contexts["out_hand"]))

    def tick(self):
        if self.is_in(self.state, "a"):
            self.in_hand = True
        elif self.rewardamount > self.rewardresetamount:
            self.in_hand = False

        PlaceCellEnvironment.tick(self)

    def update_reward(self):
        self.reward = self.defaultreward

        if self.in_hand and self.is_in(self.state, "b"):
            self.reward = 1.5
            self.rewardamount += 1

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
        # we look at whether or not we have the package to pick a goal state

        stepsize = 0.1
        self.optimal_move = None
        for y in [v * stepsize for v in
                  range(int(-self.imgsize[1] / (2 * stepsize)) + 1,
                        int(self.imgsize[1] / (2 * stepsize)) - 1)]:
            for x in [v * stepsize for v in
                      range(int(-self.imgsize[0] / (2 * stepsize)) + 1,
                            int(self.imgsize[0] / (2 * stepsize)) - 1)]:
                if ((self.is_in((x, y), "a") and not self.in_hand) or
                        (self.is_in((x, y), "b") and self.in_hand)):
                    angle = math.atan2(y - self.state[1], x - self.state[0])
                    pt = (math.cos(angle), math.sin(angle))
                    self.optimal_move = max(
                        self.actions, key=lambda x:-1
                        if self.is_in((x[1][0] * self.dx + self.state[0],
                                       x[1][1] * self.dx + self.state[1]),
                                      "wall")
                        else HRLutils.similarity(x[1], pt))[0]

                    return

    def get_image(self):
        """Generate a BufferedImage representing the current environment, for
        use in interactivemode display."""

        # copy map
        bitmap = BufferedImage(self.map.getColorModel(),
                               self.map.copyData(None), False, None)

        # draw agent
        graphics = bitmap.createGraphics()
        if self.in_hand:
            graphics.setColor(Color.green)
        else:
            graphics.setColor(Color.orange)
        agentsize = 0.2
        x, y = self.pt_to_pixel((self.state[0] - agentsize / 2,
                                 self.state[1] + agentsize / 2))
        graphics.fillRect(x, y,
                          int(agentsize * bitmap.getWidth() /
                              self.imgsize[0]),
                          int(agentsize * bitmap.getHeight() /
                              self.imgsize[1]))

        return bitmap
