def increment_dictionary(d,k):
    d[k] = d.get(k,0)+1


def slurp_lines(f):
    with open(f,'r') as fj:
        l = fj.read().splitlines()
    return l
    
def find_bad_parses():
    failures = []
    for P in range(1,24):
        positives = [ "pictures/%i_1_%i" % (P,x) for x in range(100) ]
        negatives = [ "pictures/%i_0_%i" % (P,x) for x in range(100) ]
        domain = positives + negatives
        shapes = {}
        contains = {}
        touches = {}
        for f in domain:
            l = slurp_lines(f)
            contains[f] = '\n'.join(l).count('contains')
            touches[f] = '\n'.join(l).count('borders')
            shapes[f] = l[0].count('[')
        shape_count = {}
        contains_count = {}
        touch_count = {}
        for f in domain:
            increment_dictionary(shape_count,shapes[f])
            increment_dictionary(contains_count,contains[f])
            increment_dictionary(touch_count,touches[f])
            
        correct_shape = max([(v,j) for j,v in shape_count.iteritems() ])[1]
        for f in domain:
            if shapes[f] != correct_shape or contains_count[contains[f]] < 25 or touch_count[touches[f]] < 25:
                failures.append(f)
    return failures

if __name__ == '__main__':
    print find_bad_parses()
