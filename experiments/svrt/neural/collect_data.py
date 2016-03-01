import os
import sys
s = int(sys.argv[1])
e = int(sys.argv[2])

for i in range(s,e+1):
    k = "nohup th neural.lua %i > Alex_1000/%i &" % ((i+1),(i+1))
    os.system(k)
