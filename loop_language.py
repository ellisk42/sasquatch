import sys
import os

N = int(sys.argv[1])
L = 4 #sys.argv[2]
for n in range(N):
    os.system("longjob  -o 4timing/%s python -u language.py %s lexicon" % (n+1,L))
