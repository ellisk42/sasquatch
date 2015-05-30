import os

for i in range(23):
    k = "nohup th neural.lua %i > digits_1000/%i &" % ((i+1),(i+1))
    os.system(k)
