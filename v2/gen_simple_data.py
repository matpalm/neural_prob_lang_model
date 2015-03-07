#!/usr/bin/env python
import random, sys

def label(n):
#  if n in ['Ga', 'Gb']:
#    return 'G'
  return n

TRANSISTIONS = {
 'A1': [('B1', 0.5), ('B2', 1.0)],
 'A2': [('B1', 0.5), ('B2', 1.0)],
 'B1': [('C1', 0.5), ('C2', 1.0)],
 'B2': [('C1', 0.5), ('C2', 1.0)],
 'C1': [('D1', 0.5), ('D2', 1.0)],
 'C2': [('D1', 0.5), ('D2', 1.0)],
 'D1': [('A1', 0.5), ('A2', 1.0)],
 'D2': [('A1', 0.5), ('A2', 1.0)],
}

#TRANSISTIONS = {
# 'A': [('B', 1.0)],
# 'B': [('C', 1.0)],
# 'C': [('D', 1.0)],
# 'D': [('A', 1.0)],
#}

#TRANSISTIONS = {
# 'A': [('B', 1.0)],
# 'B': [('C1', 0.5), ('C2', 1.0)],
# 'C1': [('A', 1.0)],
# 'C2': [('A', 1.0)],
#}

def next_from(n):
  trans = TRANSISTIONS[n]
  r = random.random()
  i = 0
  while r >= trans[i][1]:
    i += 1
  return trans[i][0]

n = 'A1'
for i in range(int(sys.argv[1])):
    print label(n),
    n = next_from(n)
print
 
 
