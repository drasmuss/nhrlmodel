# Copyright 2014, Daniel Rasmussen.  All rights reserved.

import nef


class DataNode(nef.SimpleNode):
    """Node to collect data and output it to file."""

    def __init__(self, period=1, dt=0.001, filename=None, header=""):
        """Initialize node variables.

        :param period: specifies how often the node should create a new data
            entry
        :param filename: name of file to save data to
        :param header: will be written to top of file (for record keeping)
        """

        nef.SimpleNode.__init__(self, "DataNode")

        self.period = period
        self.dt = dt
        self.filename = filename
        self.header = header

        self.sources = []
        self.records = []
        self.types = []

    def record(self, origin, func=lambda x: x):
        """Record data from the given origin.

        :param origin: origin to record data from
        :param func: function applied to the output of origin
        """

        self.sources += [origin]
        self.records += [[[self.t + 0.5 * self.period, None]]]
        self.types += [func]

    def record_avg(self, origin):
        self.record(origin, lambda s: [float(sum(s)) / len(s)])

    def record_sparsity(self, origin):
        self.record(origin, lambda s: [len([x for x in s if x < 0.01]) /
                                       float(len(s))])

    def tick(self):
        for i, r in enumerate(self.records):
            # get data from origin
            try:
                s = self.sources[i].getValues().getValues()
            except:
                # this can fail if the simulator is currently in the process
                # of writing to the origin
                continue

            # apply function to data
            s = self.types[i](s)
            if isinstance(s, (float, int)):
                s = [s]

            # add data to entry
            r[-1][1] = s if r[-1][1] is None else [x + y for x, y in
                                                   zip(r[-1][1], s)]

        # if period has elapsed, create a new data entry
        if self.t > 0.0 and self.t % self.period < self.dt * 1e-3:
            # divide to get avg
            num_timesteps = self.period / self.dt
            for r in self.records:
                r[-1][1] = [float(x) / num_timesteps for x in r[-1][1]]

            # write data to file
            if self.filename is not None:
                f = open(self.filename, "w")
                f.write(self.header + "\n")
                f.write("\n".join([";".join([" ".join([str(v)
                                                       for v in [entry[0]] +
                                                       entry[1]])
                                             for entry in r])
                                   for r in self.records]))
                f.close()

            # create new entry
            for r in self.records:
                r += [[self.t + 0.5 * self.period, None]]
                assert len(r) == int(self.t / self.period) + 1
