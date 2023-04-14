import json
import copy


class Logger:
    def __init__(self):
        self.log = {}

    def add(self, k, v):
        assert k not in self.log
        self.log[k] = v

    def flush(self):
        ret = copy.deepcopy(self.log)
        self.log = {}
        return ret
