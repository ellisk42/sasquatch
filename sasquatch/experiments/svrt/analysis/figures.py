from pylab import *
import math
import numpy as np
import numpy.numarray as na
from accuracies import *

rcParams.update({'figure.autolayout': True})

interactive(True)
labels = [str(i) for i in range(1,24) ]
human_failures = [1,0,0,0,4,12,4,0,3,1,0,2,3,1,2,9,9,3,1,1,7,0,0]
human_success = [float(23-x)/23.0 for x in human_failures ]
human_error = [ math.sqrt(p*(1-p)/23.0) for p in human_success ] 

program_success,program_error = [1.0, 0.5701030927835051, 0.8734536082474225, 1.0, 0.9505154639175256, 0.5175257731958763, 1.0, 0.9729381443298968, 0.9564432989690722, 0.9762886597938145, 0.9953608247422681, 0.5074742268041237, 0.7255154639175259, 0.9092783505154639, 0.9837628865979381, 0.9608247422680412, 0.5458762886597938, 0.9822164948453608, 0.9871134020618555, 0.5, 0.5, 0.9701030927835053, 0.8533505154639176], [0.0, 0.1402061855670103, 0.08431774986243508, 0.0, 0.07320495079512139, 0.0589502348977053, 0.0, 0.02706185567010311, 0.08918009938587561, 0.04462262496459618, 0.0015463917525773137, 0.028894729791332237, 0.07739901418734556, 0.11958762886597941, 0.04871134020618556, 0.002525247157508422, 0.061584456964143934, 0.006958762886597946, 2.220446049250313e-16, 0.0, 0.0, 0.08969072164948458, 0.194332118721746]


boost_success = [0.0156,0.0158,0.0532,0.0689,0.1334,0.2384,0.2408,0.0986,0.3207,0.0589,0.0586,0.1615,0.1032,0.2663,0.0041,0,0.3306,0.0092,0.3868,0.3009,0.4958,0.0307,0.2511]
#boost_success = [0.4267,0.0303,0.0462,0.0738,0.4305,0.2206,0.4679,0.1026,0.2804,0.1273,0.0415,0.1502,0.3257,0.2595,0.3143,0.3019,0.2998,0.0434,0.4924,0.4215, 0.4887, 0.3566, 0.2310]
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
                    label = 'Baseline')

axis.set_xlim(-w,len(index)+w)
axis.set_ylim(0,1)
axis.set_xticks(index+w)
axis.set_xticklabels(labels)
axis.legend((people[0],machines[0],boosting[0]),('Human (about 3 examples, varies)','Program induction (3 examples)','Boosting (10000 examples)'),loc = 3)

xlabel('SVRT Problem')
ylabel('Accuracy')

show()



savefig('comparison.png')

feature_success = [np.average(f) for f in feature_accuracies ]
program_success = [np.average(np.array(a)) for a in program_accuracies ]
accuracy_matrix = np.array([human_success,program_success,feature_success,boost_success,digit_accuracies])

print program_success
covariance = np.corrcoef(accuracy_matrix)
print [ math.sqrt(rs) for rs in covariance[0,:] ]


def scatter_frequencies(observations,color,t,x_labels,y_labels,offset = 0):
    frequencies = {}
    matrix = []
    ys = []
    for p in range(23):
        h = human_success[p]
        if isinstance(observations[p],list):
            for a in observations[p]:
                frequencies[(h,a)] = 1 + frequencies.get((h,a),0)
                matrix.append([h,a])
                ys.append(a)
        else:
            a = observations[p]
            matrix.append([h,a])
            frequencies[(h,a)] = 1 + frequencies.get((h,a),0)
            ys.append(a)
    x = [ h for ((h,a),f) in frequencies.iteritems() ]
    y = [ a for ((h,a),f) in frequencies.iteritems() ]
    area = [ f*20 for ((h,a),f) in frequencies.iteritems() ]
    covariance = np.corrcoef(np.array(matrix).T)
    r = math.sqrt(covariance[0,1])
    r = round(r,2)
    text(0.35,0.95+offset,"r=%.2f"%r,fontsize = 10)
    title(t,fontsize = 10)
    tick_places = np.arange(0.3,1.1,0.1)
    tick_names = ['0.3'] + ['']*6 + ['1']
    s = scatter(x,y,s = area,alpha = 0.5,color = color)
    if x_labels:
        xticks(tick_places,tick_names,fontsize = 10)
    else:
        xticks(tick_places,['']*7,fontsize = 10)
    if y_labels:
        yticks(tick_places,tick_names,fontsize = 10)
    else:
        yticks(tick_places,['']*7,fontsize = 10)
    a = np.average(ys)
    print a
    axhline(y = a,color = 'k',ls = 'dashed')
    a = len([y for y in ys if y >= 0.9 ])
    text(0.35,0.35,"solved %i/23" % a,fontsize = 10)
    return s

common = figure(figsize = (4,4))
subplot(221)
pi = scatter_frequencies(program_success,'r','Program synthesis',
                         False,True,-0.05)
subplot(222)
le = scatter_frequencies(digit_accuracies,'y','ConvNet',
                         False,False)
subplot(223)
b = scatter_frequencies(boost_success,'b','Image features',
                        True,True,-0.05)
subplot(224)
f = scatter_frequencies(feature_success,'g','Parse features',
                        True,False,+0.05)
#xlabel('Human accuracy')
#ylabel('Machine accuracy')
common.text(0.5,0.035,'Human accuracy',
            ha = 'center',va = 'center',fontsize = 10)
common.text(0.035,0.5,'Machine accuracy',
            ha = 'center',va = 'center',rotation = 'vertical',
            fontsize = 10)

#legend((pi,f,b,le),('Program induction (3 examples)', 'Features (3 examples)', 'Boosting (10000 examples)','LeNet (100 examples)'),loc = 'lower right')

show()
savefig('scatter.png')


figure(figsize = (2,2))


al = scatter_frequencies(alex_accuracies,'r','AlexNet variant',
                         True,True,-0.05)
xlabel('Human accuracy')
ylabel('Machine accuracy')
show()
savefig('alex.png')


figure(figsize = (2,2))
gcf().subplots_adjust(wspace=1)
le = scatter_frequencies(digit_accuracies,'y','LeNet Variant',
                         True,True)

xlabel('Human accuracy')
ylabel('Machine accuracy')
show()
savefig('lenet.png')

#xlabel('Human accuracy')
#ylabel('Machine accuracy')

#legend((pi,f,b,le),('Program induction (3 examples)', 'Features (3 examples)', 'Boosting (10000 examples)','LeNet (100 examples)'),loc = 'lower right')



