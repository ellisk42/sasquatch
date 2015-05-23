import os
import sys
import random

def dinner_party(tasks, cores = 10):
    descriptors = {} # for each running process, a file descriptor
    outputs = 0 # accumulate the number of outputs
    number_tasks = len(tasks)
    output_array = [""]*number_tasks
    next_task = 0
    
    while outputs < number_tasks:
        while len(tasks) > 0 and len(descriptors) < cores:
            # launch another task
            task = tasks[0]
            tasks = tasks[1:]
            
            r,w = os.pipe()
            p = os.fork()
            if p == 0:
                # give us a unique random seed
                random.seed(os.getpid())
                sys.stdout = os.fdopen(w, "w")
                print task(),
                sys.exit()
            else:
                descriptors[p] = (r,next_task)
                next_task += 1
        # slurp some more dinner!!
        if len(descriptors) > 0:
            p,s = os.waitpid(-1,0)
            handle,index = descriptors[p]
            o = os.read(handle,200000)
            outputs += 1
            del descriptors[p]
            output_array[index] = o

    return output_array

def make_dinner_task(f,x):
    def k():
        return f(x)
    return k

def parallel_map(f,l,cores = 10):
    return dinner_party([make_dinner_task(f,l[i]) for i in range(len(l)) ],cores = cores)




if __name__ == "__main__":
    def f(i):
        return "%i-%f" % (i,random.random())
    print parallel_map(f,list(range(10)))
