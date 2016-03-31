import sys
import os

N = int(sys.argv[1])
L = sys.argv[2]
for n in range(N):
    os.system("nohup python -u language.py %s lexicon > random_sample/%s/%s &" % (L,L,n))
