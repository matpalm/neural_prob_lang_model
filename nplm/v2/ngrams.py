#!/usr/bin/env python
import sys
if len(sys.argv) != 2:
    print >>sys.stderr, "usage: %s NGRAMSIZE" % sys.argv[0]
    exit(1)
N=int(sys.argv[1])
for line in sys.stdin:
    tokens = line.strip().split(" ")
    for i in xrange(len(tokens)-N+1):
        print " ".join(tokens[i:i+N])

    
