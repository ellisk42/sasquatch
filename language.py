from solverUtilities import *
from z3 import *
import math
import random

TENSES = 2

# map between tipa and z3 character code
ipa2char = { 'p': 'Pp', 'b': 'Pb', 'm': 'Pm', 'f': 'Pf', 'v': 'Pv', 'T': 'PT', 'D': 'PD', 'R': 'PR', 't': 'Pt', 'd': 'Pd',
             'n': 'Pn', 'r': 'Pr', 's': 'Ps', 'z': 'Pz', 'l': 'Pl', 'S': 'PS', 'Z': 'PZ', 'j': 'Pj', 'k': 'Pk', 'w': 'Pw',
             'g': 'Pg', 'N': 'PN', 'P': 'PP', 'h': 'Ph', 'i': 'Pi', 'I': 'PI', 'e': 'Pe', 'E': 'PE', '\\ae': 'PQ', '@': 'P@',
             '2': 'P2', 'A': 'PA', 'a': 'Pa', '5': 'P5', '0': 'P0', 'o': 'Po', 'U': 'PU', 'u': 'Pu'
             }
char2ipa = {}
for k in ipa2char: char2ipa[ipa2char[k]] = k

Phoneme, phonemes = EnumSort('Phoneme', tuple(char2ipa.keys()))
z3char = {}
for p in phonemes:
    z3char[str(p)] = p

# maximum string length
maximum_length = 9


def morpheme():
    l = integer()
    constrain(l < maximum_length+1)
    constrain(l > -1)
    ps = [ Const(new_symbol(), Phoneme) for j in range(maximum_length) ]
    return tuple([l]+ps)


def extract_string(m, v):
    rv = ""
    l = m[v[0]].as_long()
    ps = list(v)[1:]
    for j in range(l):
        c = str(m[ps[j]])
        cp = char2ipa[c]
        if cp[0] == '\\':
            rv += cp + " "
        else:
            rv += cp
    return "\\textipa{%s}" % rv

def constrain_phonemes(ps,correct):
    l = ps[0]
    ps = list(ps)[1:]
    correct = correct.split(' ')
    constrain(l == len(correct))
    
    correct = [ z3char[ipa2char[c]] for c in correct ]
    assert(len(correct) < maximum_length+1)
    for j in range(len(correct)):
        constrain(ps[j] == correct[j])

def concatenate(p,q):
    r = morpheme()
    lr = r[0]
    r = list(r)[1:]
    lp = p[0]
    p = list(p)[1:]
    lq = q[0]
    q = list(q)[1:]
    
    constrain(lr == lp+lq)
    constrain(lr < maximum_length+1)
    
    for j in range(maximum_length):
        constrain(Implies(lp > j,
                          r[j] == p[j]))
        constrain(Implies(lp == j,
                          And(*[ Implies(lq > i, r[i+j] == q[i])
                                 for i in range(maximum_length-j) ])))
    return tuple([lr]+r)
                          
                          


def last_one(ps):
    constrain(ps[0] > 0)
    ending = Const(new_symbol(), Phoneme)
    for j in range(1,maximum_length+1):
        constrain(Implies(ps[0] == j,ending == ps[j]))
    return ending


def voiced(p):
    targets = 'bmvDRdnzZjlgNiIeE@2Aa50oUu'
    targets = ['\\ae'] + [t for t in targets ]
    v = boolean()
    constrain(v == Or(*[ p == z3char[ipa2char[t]] for t in targets ]))
    return v

def primitive_string():
    thing = morpheme()
    def evaluate_string(i):
        return thing
    def print_string(m):
        return extract_string(m,thing)
    m = real()
    constrain(m == logarithm(44)*thing[0])
    return evaluate_string, m, print_string

indexed_rule('STEM', 'stem', TENSES,
             lambda i: i)
rule('VOICED',['STEM'],
     lambda m,st: "(voiced? (last-one %s))" % st,
     lambda i,st: voiced(last_one(st)))
rule('PREDICATE',['VOICED'],
     lambda m,v: v,
     lambda i,v: v)
rule('RETURN',['STEM','STRING'],
     lambda m, stem, suffix: stem if suffix == '\\textipa{}' else "(append %s %s)" % (stem,suffix),
     lambda i, p, q: concatenate(p,q))
rule('CONDITIONAL',['PREDICATE','RETURN','CONDITIONAL'],
     lambda m, p,q,r: "(if %s %s %s)" % (p,q,r),
     lambda i, p,q,r: conditional(p,q,r))
rule('CONDITIONAL',['RETURN'],
     lambda m, r: r,
     lambda i, r: r)
primitive_rule('STRING',
               primitive_string)

# for each tense, a different rule
programs = [ generator(2,'CONDITIONAL') for j in range(TENSES) ]

observations = [ [ "b a l z", "b a l"],
                 ["d o g z", "d o g"],
                 ["r u n z","r u n"],
                 ["i t s","i t"],
                 ["w a k s","w a k"]]*1
N = len(observations)

stems = [ [morpheme() for i in range(TENSES)] for j in range(N) ]

for t in range(TENSES):
    for n in range(N):
        o = programs[t][0](stems[n])
        constrain_phonemes(o, observations[n][t])

def printer(m):
    
    model = ""
    for t in range(TENSES):
        model += "Tense %i: %s\n" % (t,programs[t][2](m))
    model += "\n"
    for j in range(N):
        model += "\t".join(["stem[%i] = %s" % (t,extract_string(m,stems[j][t])) for  t in range(TENSES) ])
        model += "\n"
    return model

flat_stems = [ v for sl in stems for v in sl  ]
total = summation([p[1] for p in programs ] + [logarithm(44)*s[0] for s in flat_stems ])


compressionLoop(printer,total)

