from pylab import *
import math
import numpy as np
import numpy.numarray as na

interactive(True)
labels = [str(i) for i in range(1,24) ]
human_failures = [1,0,0,0,4,12,4,0,3,1,0,2,3,1,2,9,9,3,1,1,7,0,0]
human_success = [float(23-x)/23.0 for x in human_failures ]
human_error = [ math.sqrt(p*(1-p)/23.0) for p in human_success ] 

program_success,program_error = [1.0, 0.5700000000000001, 0.8037234042553191, 1.0, 0.9481481481481481, 0.5954773869346733, 1.0, 0.8939393939393939, 0.9518324607329843, 0.9890000000000001, 0.9798994974874372, 0.4887755102040816, 0.8704819277108434, 0.9693877551020409, 1.0, 0.9947368421052631, 0.53, 0.5, 0.5, 0.5, 0.5, 0.9443298969072165, 0.9556122448979592], [0.0, 0.13999999999999999, 0.03705885840074544, 0.0, 0.10370370370370371, 0.08657228328586145, 0.0, 0.0, 0.058992946684830454, 0.013472193585307464, 0.0, 0.041998195157666694, 0.05550674457729852, 1.1102230246251565e-16, 0.0, 0.0, 0.03674234614174765, 0.0, 0.0, 0.0, 0.0, 0.11134020618556702, 0.03624245027587358]

boost_success = [0.4267,0.0303,0.0462,0.0738,0.4305,0.2206,0.4679,0.1026,0.2804,0.1273,0.0415,0.1502,0.3257,0.2595,0.3143,0.3019,0.2998,0.0434,0.4924,0.4215, 0.4887, 0.3566, 0.2310]
boost_success = [1-x for x in boost_success ]
boost_error = [0.0]*23

fig,axis = subplots()

w = 0.28
index = na.arange(len(labels))

people = axis.bar(index,
                  human_success,yerr = human_error,
                  width = w,
                  label = 'Human')
machines = axis.bar(index+w,
                    program_success,yerr = program_error,
                    width = w,
                    color = 'r',
                    label = 'PI')
boosting = axis.bar(index+w+w,
                    boost_success, #yerr = boost_error,
                    width = w,
                    color = 'g',
                    label = 'Boosting')

axis.set_xlim(-w,len(index)+w)
axis.set_ylim(0,1)
axis.set_xticks(index+w)
axis.set_xticklabels(labels)
axis.legend((people[0],machines[0],boosting[0]),('Human (about 3 examples, varies)','Program induction (3 examples)','Boosting (10000 examples)'),loc = 3)
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

accuracy_matrix = np.array([human_success,program_success,boost_success])
covariance = np.corrcoef(accuracy_matrix)
print covariance
