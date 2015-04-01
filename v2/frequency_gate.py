import time

# this is a dumb name... what should this be called?
class FrequencyGate(object):

    def __init__(self, open_delay):
        self.open_delay = open_delay
        self.last_release = time.time()

    def open(self):
        t = time.time()
        if t - self.last_release >= self.open_delay:
            self.last_release = t
            return True
        else:
            return False
        
