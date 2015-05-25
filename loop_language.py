import sys
import os

N = int(sys.argv[1])
L = int(sys.argv[2])
for n in range(N):
    os.system("python language.py %i coupled" % L)
