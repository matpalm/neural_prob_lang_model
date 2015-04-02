# utils to read/write a set of np arrays over time

import glob
import re
import time
import numpy as np

# TODO: the need for this method is a hack, encoding this info into the filename is clumsy...
def prefix_time_var_of_ckpt(ckpt_filename):
    m = re.match("(.*?)\.(\d+)\.(.*)", ckpt_filename)
    if m: return tuple(m.groups())
    raise Exception("malformed ckpt_filename? [" + ckpt_filename + "]")

class Checkpointer(object):

    # TODO: is there any way to get name of variable (for filename) from a reference?
    #       would like var_names to be automatic...
    def __init__(self, working_dir, space_seperated_list_of_var_names):
        self.prefix = working_dir + "/ckpt"
        self.var_names = space_seperated_list_of_var_names.split()
    
    def latest_checkpoint(self):
        """ return latest checkpoint """
        latest = None
        for candidate in glob.glob(self.prefix + "*"):
            m = re.match("^" + self.prefix + "\.(\d+)\.", candidate)
            if m: latest = max(latest, int(m.group(1)))
        return latest

    def load_checkpoint(self, checkpoint):
        """ returns an [] of vars, one per element from var_names """
        return [np.load("%s.%d.%s" % (self.prefix, checkpoint, vn)) for vn in self.var_names]

    def dump_checkpoint(self, vs):
        """ write vars in 'vs' to files based on names from 'var_names' """
        if len(vs) != len(self.var_names):
            raise Exception("given %s variables to dump by configured with a different number of vars %s" % (len(vs), self.var_names))
        checkpoint = int(time.time())
        for var_name, var in zip(self.var_names, vs):
            var.dump("%s.%d.%s" % (self.prefix, checkpoint, var_name))
                
               

