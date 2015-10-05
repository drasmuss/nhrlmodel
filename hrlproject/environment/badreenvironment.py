# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.util import MU
from ca.nengo.util.impl import RandomHypersphereVG

from hrlproject.environment.environmenttemplate import EnvironmentTemplate
from hrlproject.misc import HRLutils
from hrlproject.misc.HRLutils import rand as random


class BadreEnvironment(EnvironmentTemplate):
    """Environment to recreate the task from Badre et al. (2010) 'Frontal
    cortex and the discovery of abstract action rules.'

    :input action: vector representing action selected by agent
    :output state: vector representing current state
    :output reward: reward value
    """

    def __init__(self, flat=False):
        """Set up task parameters.

        :param flat: if True, no hierarchical relationship between stimuli and
            reward; if False, stimuli-response rewards will be dependent on
            colour
        """

        self.rewardval = 1.5

        # actions correspond to three different button presses
        actions = [("left", [1, 0, 0]), ("middle", [0, 1, 0]),
                   ("right", [0, 0, 1])]

        # number of instances of each attribute (stimuli formed through
        # different combinations of attribute instances)
        self.num_orientations = 3
        self.num_shapes = 3
        self.num_colours = 2

        self.presentationtime = 0.5  # length of time to present each stimuli
        self.rewardtime = 0.1  # length of reward period

        # next presentation interval
        self.presentationperiod = [0, self.presentationtime]

        # next reward interval
        self.rewardperiod = [self.presentationtime,
                             self.presentationtime + self.rewardtime]

        self.answer = random.choice(actions)[0]  # answer selected by agent

        self.stateD = (self.num_orientations + self.num_shapes +
                       self.num_colours)

        self.correct = [0] * 20

        EnvironmentTemplate.__init__(self, "BadreEnvironment", self.stateD,
                                     actions)

        self.answers = self.gen_answers(flat)

        self.create_origin("optimal_move",
                           lambda: [a[1] for a in actions
                                    if a[0] == self.answer][0])
        self.create_origin("score",
                           lambda: [sum(self.correct) / len(self.correct)])

    def tick(self):
        """Update state/reward each timestep."""

        # present stimuli
        if (self.t > self.presentationperiod[0] and
                self.t < self.presentationperiod[1] and
                self.state == [0 for _ in range(self.stateD)]):
            # pick a random stimuli at beginning of presentation period
            # and set that as the current state for the duration of
            # the presentation period
            self.state = random.choice(self.answers.keys())
            self.answer = self.answers[self.state]
            self.state = list(self.state)

        # provide feedback if in reward period
        if self.t > self.rewardperiod[0] and self.t < self.rewardperiod[1]:
            self.reward = (self.rewardval if self.action[0] == self.answer
                           else -self.rewardval)
        else:
            self.reward = 0

        # update score
        if ((self.t + self.rewardtime) % (self.presentationtime +
                                          self.rewardtime)) < 0.002:
            self.correct = self.correct[1:] + ([1.0] if self.action[0] ==
                                               self.answer else [0.0])

        # update presentation/reward period
        if (self.t % (self.presentationtime + self.rewardtime)) < 0.002:
            self.presentationperiod = [self.t, self.t + self.presentationtime]
            self.rewardperiod = [self.t + self.presentationtime, self.t +
                                 self.presentationtime + self.rewardtime]
            self.state = [0 for _ in range(self.stateD)]

    def gen_answers(self, flat):
        """Generate the stimuli-response mappings to be used in the task."""

        # each instance is a unit vector with dimension equal to the number of
        # instances
        orientations = [[1 if i == j else 0
                         for i in range(self.num_orientations)]
                        for j in range(self.num_orientations)]
        shapes = [[1 if i == j else 0 for i in range(self.num_shapes)]
                  for j in range(self.num_shapes)]
        colours = [[1 if i == j else 0 for i in range(self.num_colours)]
                   for j in range(self.num_colours)]

        # stimuli are formed from all possible combinations of attribute
        # instances
        stimuli = []
        for o in orientations:
            for s in shapes:
                for c in colours:
                    stimuli += [o + s + c]

        answers = {}

        if flat:
            # just pick a random response for each stimulus
            responses = (["left" for _ in range(len(stimuli) / 3)] +
                         ["middle" for _ in range(len(stimuli) / 3)] +
                         ["right" for _ in range(len(stimuli) / 3)])
            random.shuffle(responses)
            for stim in stimuli:
                answers[tuple(stim)] = responses.pop()
        else:
            # response depends on the identity of the colour attribute
            for stim in stimuli:
                if stim[-self.num_colours:] == [0, 1]:
                    # then we care about shape
                    answers[tuple(stim)] = self.actions[
                        shapes.index(stim[self.num_orientations:
                                          - self.num_colours])][0]
                else:
                    # then we care about orientation
                    answers[tuple(stim)] = self.actions[
                        orientations.index(stim[:self.num_orientations])][0]

        return answers

    def gen_encoders(self, N, contextD, context_scale):
        """Generate encoders for state population of learning agent.

        :param N: number of neurons in state population
        :param contextD: dimension of context vector representation
        :param context_scale: weight on context representation relative to
            state (1.0 = equal weighting)
        """

        if contextD > 0:
            contexts = MU.I(contextD)
        else:
            contexts = [[]]

        # neurons each sensitive to different combinations of stimuli
        encs = (list(MU.I(self.stateD)) +
                [o + s + c
                 for o in MU.I(self.num_orientations)
                 for s in MU.I(self.num_shapes)
                 for c in MU.I(self.num_colours)])

        return [HRLutils.normalize(
            HRLutils.normalize(random.choice(encs)) +
            [x * context_scale for x in random.choice(contexts)])
            for _ in range(N)]

        # random encoders
#         encoders = RandomHypersphereVG().genVectors(N, self.num_orientations +
#                                                     self.num_shapes +
#                                                     self.num_colours)
#         if contextD > 0:
#             contexts = MU.diag([context_scale for _ in range(contextD)])
#             encoders = [HRLutils.normalize(e + random.choice(contexts))
#                         for e in encoders]
#
#         return encoders
