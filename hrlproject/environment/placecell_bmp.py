# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import math

from javax.imageio import ImageIO
from java.io import File
from java.awt import Color
from java.awt.image import BufferedImage

from hrlproject.environment.environmenttemplate import EnvironmentTemplate
from hrlproject.misc import HRLutils
from hrlproject.misc.HRLutils import rand as random


class PlaceCellEnvironment(EnvironmentTemplate):
    """An environment that represents the agent's location in continuous space
    through simulated place cell activations.

    :input action: vector representing action selected by agent
    :output state: current x,y location of agent
    :output reward: reward value
    :output place: vector concatenating activation of each place cell
    :output optimal_move: the ideal action for the agent to take in the
        current state
    """

    def __init__(self, actions, mapname, colormap, name="PlaceCellEnvironment",
                 imgsize=(1.0, 1.0), dx=0.01, placedev=0.1, num_places=None):
        """Initialize environment variables.

        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param mapname: name of file describing environment map
        :param colormap: dict mapping pixel colours to labels
        :param name: name for environment
        :param imgsize: width of space represented by the map image
        :param dx: distance agent moves each timestep
        :param placedev: standard deviation of gaussian place cell activations
        :param num_places: number of placecells to use (if None it will attempt
            to fill the space)
        """

        EnvironmentTemplate.__init__(self, name, 2, actions)

        # parameters
        self.colormap = colormap
        self.rewardamount = 0  # number of timesteps spent in reward

        # number of timesteps to spend in reward before agent is reset
        # note: convenient to express this as time_in_reward / dt
        self.rewardresetamount = 0.6 / 0.001

        self.num_actions = len(actions)
        self.imgsize = [float(x) for x in imgsize]
        self.dx = dx
        self.placedev = placedev
        self.num_places = num_places
        self.optimal_move = None
        self.defaultreward = -0.075

        # load environment
        self.map = ImageIO.read(File(HRLutils.datafile(mapname)))

        # generate place cells
        self.gen_placecells(min_spread=1.0 * placedev)

        # initial conditions
        self.state = self.random_location(avoid=["wall", "target"])
        self.place_activations = [0 for _ in self.placecells]

        self.create_origin("place", lambda: self.place_activations)

        # note: making the value small, so that the noise node will give us
        # some random exploration as well
        self.create_origin("optimal_move",
                           lambda: [1.0 if self.optimal_move == a[0] else 0.0
                                    for a in self.actions])

    def tick(self):
        self.update_state()

        # update place cell activations
        dists = [self.calc_dist(self.state, l) for l in self.placecells]
        self.place_activations = [math.exp(-d ** 2 / (2 * self.placedev ** 2))
                                  for d in dists]

        self.update_reward()

        self.calc_optimal_move()

    def update_state(self):
        dest = self.state

        if self.action is not None:
            if self.action[0] == "up":
                dest = [self.state[0], self.state[1] + self.dx]
            elif self.action[0] == "right":
                dest = [self.state[0] + self.dx, self.state[1]]
            elif self.action[0] == "down":
                dest = [self.state[0], self.state[1] - self.dx]
            elif self.action[0] == "left":
                dest = [self.state[0] - self.dx, self.state[1]]
            else:
                print "Unrecognized action"

        if not self.is_in(dest, "wall"):
            self.state = dest

        self.dest = dest

        # reset location if been in reward long enough
        if self.rewardamount > self.rewardresetamount:
            self.state = self.random_location(avoid=["wall", "target"])
            self.rewardamount = 0

    def update_reward(self):
        if self.is_in(self.state, "target"):
            self.reward = 1

            self.rewardamount += 1
        else:
            self.reward = self.defaultreward

    def random_location(self, avoid=[]):
        """Pick a random location, avoiding regions with the specified
        labels."""

        pt = (random.uniform(-self.imgsize[0] / 2.0, self.imgsize[0] / 2.0),
              random.uniform(-self.imgsize[1] / 2.0, self.imgsize[1] / 2.0))

        while any([self.is_in(pt, s) for s in avoid]):
            pt = (random.uniform(-self.imgsize[0] / 2.0,
                                 self.imgsize[0] / 2.0),
                  random.uniform(-self.imgsize[1] / 2.0,
                                 self.imgsize[1] / 2.0))

        return pt

    def pt_to_pixel(self, pt):
        """Convert a pt in x,y space to a pixel in the map image."""

        x, y = pt
        width = float(self.map.getWidth())
        height = float(self.map.getHeight())

        # shift pt over into first quadrant
        x += self.imgsize[0] / 2.0
        y = -y + self.imgsize[1] / 2.0

        # convert to pixel location
        x = int(x * width / self.imgsize[0])
        y = int(y * height / self.imgsize[1])

        return x, y

    def is_in(self, pt, label):
        """Returns true if the point is in a region with the given label."""

        x, y = self.pt_to_pixel(pt)

        return self.colormap[self.map.getRGB(x, y)] == label

    def calc_dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def calc_optimal_move(self):
        """Calculates the optimal move for the agent to make in the current
        state.

        Used for debugging.
        """

        # grid search the image with the given stepsize
        stepsize = 0.1
        self.optimal_move = None
        for y in [v * stepsize for v in
                  range(int(-self.imgsize[1] / (2 * stepsize)) + 1,
                        int(self.imgsize[1] / (2 * stepsize)) - 1)]:
            for x in [v * stepsize for v in
                      range(int(-self.imgsize[0] / (2 * stepsize)) + 1,
                            int(self.imgsize[0] / (2 * stepsize)) - 1)]:
                # if the pt you're looking at is in the region you're
                # looking for
                if self.is_in((x, y), "target"):
                    # generate a target point in the direction from current
                    # location to target
                    angle = math.atan2(y - self.state[1], x - self.state[0])
                    pt = (math.cos(angle), math.sin(angle))

                    # pick the action that is closest to the target point
                    # note: penalize actions that would involve moving through
                    # a wall
                    self.optimal_move = max(
                        self.actions, key=lambda x:-1
                        if self.is_in((x[1][0] * self.dx + self.state[0],
                                       x[1][1] * self.dx + self.state[1]),
                                      "wall")
                        else HRLutils.similarity(x[1], pt))[0]
                    return

    def gen_placecells(self, min_spread=0.2):
        """Generate the place cell locations that will give rise to the state
        representation.

        :param min_spread: the minimum distance between place cells
        """

        N = self.num_places

        # a limit on the number of attempts to place a new placecell
        num_tries = 1000

        # assign random x,y locations to each neuron
        locations = [self.random_location(avoid=["wall"])]
        while True:
            # generate a random new point
            new_loc = self.random_location(avoid=["wall"])

            # check that the point isn't too close to previous points
            count = 0
            while min([self.calc_dist(new_loc, l)
                       for l in locations]) < min_spread and count < num_tries:
                new_loc = self.random_location(avoid=["wall"])
                count += 1

            # add the new point
            locations += [new_loc]

            if (N is None and count >= num_tries) or len(locations) == N:
                # stop when required number of place cells built (if N
                # specified), or when world has been filled
                break

        self.placecells = locations

    def gen_encoders(self, N):
        """Generate encoders for state population in RL agent."""

        locs = self.placecells

        encoders = [None for _ in range(N)]
        for i in range(N):
            # pick a random point for the neuron
            # note: could make this avoid walls if we want
            pt = self.random_location()

            # set the encoder to be the inverse of the distance from each
            # placecell to that point
            encoders[i] = [1.0 / self.calc_dist(pt, l) for l in locs]

            # cut off any values below a certain threshold
            encoders[i] = [x if x > 0.5 * max(encoders[i]) else 0
                           for x in encoders[i]]

            # normalize the encoder
            encoders[i] = [x / math.sqrt(sum([y ** 2 for y in encoders[i]]))
                           for x in encoders[i]]

        return encoders

    def get_image(self):
        """Generate a BufferedImage representing the current environment, for
        use in interactivemode display."""

        # copy map
        bitmap = BufferedImage(self.map.getColorModel(),
                               self.map.copyData(None), False, None)

        # draw agent
        graphics = bitmap.createGraphics()
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
