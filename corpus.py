import random
from lexicon import *

tense_name = ['PRESENT','PAST','3rdSING','PASTPAR','PROG']

def sample_corpus(n,v = None):
    if v == None:
        v = celex_inflections
    s = random.sample(v,n)
    observations = []
    for r in s:
        for t in range(5):
            observations.append((r[0],t,r[t]))
    return observations

def split_lexicon():    
    s = random.getstate()
    random.seed(0)
    training = random.sample(celex_inflections,len(celex_inflections)/2)
    test = [i for i in celex_inflections if not i in training ]
    random.setstate(s)
    return training,test

def latexTable(observations):
    table = {}
    for stem,inflection,inflected in observations:
        table[stem] = table.get(stem , {})
        table[stem][inflection] = inflected
    for stem in table:
        for t in range(5):
            if t in table[stem]:
                print "&\\textipa{%s}" % table[stem][t],
            else:
                print "&",
        print "\\\\"


minimal_pairs = [#["p e","p e d","p e I N","p e d","p e","p e z"],
                 ["s e","s E d","s e I N","s E d","s e","s E z"],
                 ["w e t","w e t @ d","w e t I N","w e t @ d","w e t","w e t s"],
                 ["k I k","k I k t","k I k I N","k I k t","k I k","k I k s"],
                 ["E n d","E n d @ d","E n d I N","E n d @ d","E n d","E n d z"],
                 ["b \\ae n","b \\ae n d","b \\ae n I N","b \\ae n d","b \\ae n","b \\ae n z"],
                 #["p a p","p a p t","p a p I N","p a p t","p a p","p a p s"],
                 ["p U S","p U S t","p U S I N","p U S t","p U S","p U S @ z"]]
minimal_pairs = [ (w[0],t,w[t]) for w in minimal_pairs for t in range(6) ]

albright = ["b a j z",
            "d a j z",
            "d r a j s",
            "f l I d g",
            "f r o",
            "g E r",
            "g l I p",
            "r a j f",
            "s t I n",
            "s t I p"]

unsupervised = ["n i d @ d","w @ r k t","p l e d","w e t @ d","\\ae d @ d","E n d @ d","\\ae k t @ d","p a p t","r 0 r d"]
unsupervised_matrix = [(w,1,w) for w in unsupervised ]


def sparse_lexicon(N,t = None):
    if t == None:
        return sparse_lexicon(N,0)+sparse_lexicon(N,1)+sparse_lexicon(N,2)+sparse_lexicon(N,5)
    observations = []
    while len(observations) < N:
        w = random.sample(celex_inflections,1)[0]
        o = (w[0],t,w[t])
        if not o in observations:
            observations.append(o)
    for (s,t,o) in sorted(observations):
        print "[%s]\t%i\t[%s]" % (s,t,o)
    return observations

def coupled_sparsity(N):
    s = random.sample(celex_inflections,N)
    observations = []
    table = ''
    for r in s:
        ts = random.sample(possible_trances,2)
        for t in ts:
            observations.append((r[0],t,r[t]))
        for t in range(5):
            if t in ts:
                table += '%s&' % r[t]
            else:
                table += '&'
        table += '\\\\\n'
    for t in possible_trances:
        if not any([t == tp for r,tp,rp in observations ]):
            return coupled_sparsity(N)
    print table
    return observations
    
        

if __name__ == "__main__":
    for inflections in celex_inflections:
#        if inflections[0] == inflections[4]: continue
#        print inflections
        for inflection in inflections:
            print ''.join(inflection.replace('\\ae','Q').split(' '))
#    for (s,t,o) in sorted(sparse_lexicon(30)):
#        print "[%s]\t%i\t[%s]" % (s,t,o)
