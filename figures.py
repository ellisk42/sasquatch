from pylab import *
import math
import numpy as np
import numpy.numarray as na

interactive(True)
labels = [str(i) for i in range(1,24) ]
human_failures = [1,0,0,0,4,12,4,0,3,1,0,2,3,1,2,9,9,3,1,1,7,0,0]
human_success = [float(23-x)/23.0 for x in human_failures ]
human_error = [ math.sqrt(p*(1-p)/23.0) for p in human_success ] 

program_success = [0.5]*23
program_error = [0.1]*23

fig,axis = subplots()

w = 0.35
index = na.arange(len(labels))

people = axis.bar(index,
                  human_success,yerr = human_error,
                  width = w,
                  label = 'Human')
machines = axis.bar(index+w,
                    program_success,yerr = program_error,
                    width = w,
                    color = 'r',
                    label = 'AI')

axis.set_xlim(-w,len(index)+w)
axis.set_ylim(0,1)
axis.set_xticks(index+w)
axis.set_xticklabels(labels)
axis.legend((people[0],machines[0]),('Human','AI'))
#axis.bar(xl+w,program_success,yerr = program_error,width = w,color = 'r')
#yticks([float(y)/10.0 for y in range(11) ])
#xticks(index + w,labels)
#axis.set_xlim(0,xl[-1]+w*2)
#axis.set_ylim(0,1)
#gca().get_xaxis().tick_bottom()
#gca().get_yaxis().tick_left()

xlabel('SVRT Problem')
ylabel('Accuracy')

show()



savefig('test.png')
