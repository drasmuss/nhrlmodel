# Copyright 2014, Daniel Rasmussen.  All rights reserved.

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.util import MU

import nef

from hrlproject.misc import HRLutils


class Memory(NetworkImpl):
    """A network to store a given value on command.

    :input target: the value to be stored
    :input transfer: if ~1, change the stored value to the current value of
        target
    :output X: stored value
    """

    def __init__(self, name, N, d, radius=1.0, inputscale=1.0, recurweight=1.0,
                 direct_storage=False):
        """Builds the Memory network.

        :param name: name of network
        :param N: base number of neurons
        :param d: dimension of stored value
        :param radius: radius of stored value
        :param inputscale: controls how fast the stored value moves to the
            target
        :param recurweight: controls the preservation of the stored value
        :param direct_storage: if True, use directmode for the memory
        """

        self.name = name
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)
        self.dimension = d
        self.radius = radius

        tauPSC = 0.007
        intPSC = 0.1

        # population that will store the value
        if not direct_storage:
            storage = net.make_array("storage", N, d,
                                     node_factory=HRLutils.node_fac(),
                                     eval_points=[[x * 0.001]
                                                  for x in range(-1000, 1000)])
        else:
            storage = net.make("storage", 1, d, mode="direct")
            storage.fixMode()

        net.connect(storage, storage, transform=MU.diag([recurweight
                                                         for _ in range(d)]),
                    pstc=intPSC)

        # storageinput will represent (target - stored_value), which when used
        # as input to storage will drive the stored value to target
        storageinput = net.make_array("storageinput", N, d,
                                      node_factory=HRLutils.node_fac())
        storageinput.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        storageinput.addDecodedTermination("target",
                                           MU.diag([1.0 / radius
                                                    for _ in range(d)]),
                                           tauPSC, False)
        # note: store everything in -1 -- 1 range by dividing by radius

        # scale storageinput value by inputscale to control rate at which
        # it moves to the target
        net.connect(storageinput, storage, pstc=intPSC,
                    transform=MU.diag([inputscale * intPSC for _ in range(d)]))

        # subtract currently stored value
        net.connect(storage, storageinput, pstc=tauPSC,
                    transform=MU.diag([-1 for _ in range(d)]))

        # we want to open the input gate when the transfer signal arrives (to
        # transfer storageinput to storage). using a double inhibition setup
        # (rather than just feeding it e.g. the the inverse of the transfer
        # signal) so that we get a nice clean zero

        # this inhibits the storageinput population (to block input to the
        # storage)
        transferinhib = net.make("transferinhib", N, 1,
                                 node_factory=HRLutils.node_fac())
        transferinhib.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])

        transferinhib.addTermination("gate",
                                     [[-10] for _ in
                                      range(transferinhib.getNeurons())],
                                     tauPSC, False)

        net.connect(transferinhib, storageinput, pstc=tauPSC,
                    transform=[[-10] for _ in
                               range(storageinput.getNeurons())])

        # this drives the transferinhib population (so that by default it will
        # block any input). inhibiting transferinhib will thus remove the
        # inhibition on storageinput, and change the stored value
        biasinput = net.make_input("biasinput", [1])

        net.connect(biasinput, transferinhib, pstc=tauPSC)

        # output population (to undo radius scaling)
        storageoutput = net.make("storageoutput", 1, d, mode="direct")
        storageoutput.fixMode()
        net.connect(storage, storageoutput, pstc=0.001,
                    transform=MU.diag([radius for _ in range(d)]))

        self.exposeTermination(transferinhib.getTermination("gate"),
                               "transfer")
        self.exposeTermination(storageinput.getTermination("target"), "target")
        self.exposeOrigin(storageoutput.getOrigin("X"), "X")

    def addDecodedOrigin(self, name, funcs, origin):
        net = nef.Network(self)

        o = self.getNode("storage").addDecodedOrigin(name, funcs, origin)

        # undo radius scaling
        funcout = net.make(name, 1, self.dimension, mode="direct")
        funcout.fixMode()
        net.connect(o, funcout, pstc=0.001,
                    transform=MU.diag([self.radius for _ in
                                       range(self.dimension)]))

        self.exposeOrigin(funcout.getOrigin("X"), name)
        return self.getOrigin(name)

    def addDecodedTermination(self, name, transform, pstc, mod):
        # note, putting this termination directly onto the storage ensemble
        # (not gated)
        t = self.getNode("storage").addDecodedTermination(name, transform,
                                                          pstc, mod)

        self.exposeTermination(t, name)
        return self.getTermination(name)
