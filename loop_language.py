import sys
import os

N = int(sys.argv[1])
L = sys.argv[2]
for n in range(N):
    os.system("longjob  -o random_sample/2%s/%s python language.py %s lexicon" % (L,n,L))
