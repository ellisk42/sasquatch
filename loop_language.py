import sys
import os

N = int(sys.argv[1])
L = sys.argv[2]
for n in range(N):
    os.system("longjob  -o random_sample/%s/%s python -u language.py %s lexicon" % (L,n,L))
