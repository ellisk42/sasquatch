from solverUtilities import *
from z3 import *
import math
import random
import sys

from corpus import *

# exception probability
epsilon = 0.1


# map between tipa and z3 character code
ipa2char = { 'p': 'Pp', 'b': 'Pb', 'm': 'Pm', 'f': 'Pf', 'v': 'Pv', 'T': 'PT', 'D': 'PD', 'R': 'PR', 't': 'Pt',
             'd': 'Pd', 'n': 'Pn', 'r': 'Pr', 's': 'Ps', 'z': 'Pz', 'l': 'Pl', 'S': 'PS', 'Z': 'PZ', 'j': 'Pj',
             'k': 'Pk', 'w': 'Pw', 'g': 'Pg', 'N': 'PN', 'P': 'PP', 'h': 'Ph', 'i': 'Pi', 'I': 'PI', 'e': 'Pe',
             'E': 'PE', '@': 'P@', '2': 'P2', 'A': 'PA', 'a': 'Pa', '5': 'P5', 'O': 'PO', 'o': 'Po', 'U': 'PU',
             'u': 'Pu', '\\ae': 'PQ'
             }
char2ipa = {}
for k in ipa2char:
    assert not (ipa2char[k] in char2ipa)
    char2ipa[ipa2char[k]] = k

Phoneme, phonemes = EnumSort('Phoneme', tuple(char2ipa.keys()))
z3char = {}
for p in phonemes:
    assert not str(p) in z3char
    z3char[str(p)] = p

Place, places = EnumSort('Place', ('NoPlace','LABIAL','CORONAL','DORSAL'))
place_table = { 'LABIAL': 'p b f v m w',
                'CORONAL': 'r t d T D s z S Z n l',
                'DORSAL': 'k g h j N' }
Voicing, voices = EnumSort('Voice', ('VOICED','UNVOICED'))
voice_table = { 'VOICED': 'b m v D R r d w n z Z j l g N i I e E @ 2 A a 5 O o U u \\ae',
                'UNVOICED': 'p f T t s S P h k'}
Manner, manners = EnumSort('Manner', ('NoManner','STOP','FRICATIVE','NASAL','LIQUID','GLIDE'))
manner_table = { 'STOP': 'p b t d k g',
                 'FRICATIVE': 'f v T D s z Z S h',
                 'NASAL': 'm n N',
                 'LIQUID': 'l r',
                 'GLIDE': 'j w'}
Sibilant, sibilance = EnumSort('Sibilant', ('NoSibilant','SIBILANT'))
sibilant_table = { 'SIBILANT': 's z S Z'}

# maximum string length
# this will get overwritten, so the number nine is completely arbitrary
maximum_length = 9


def morpheme():
    l = integer()
    constrain(l > -1)
    ps = [ Const(new_symbol(), Phoneme) for j in range(maximum_length) ]
    return tuple([l]+ps)


def extract_string(m, v, tipa = True):
    rv = ""
    l = m[v[0]].as_long()
    ps = list(v)[1:]
    for j in range(l):
        c = str(m[ps[j]])
        cp = char2ipa[c]
        if cp[0] == '\\' or not tipa:
            rv += cp + " "
        else:
            rv += cp
    if tipa:
        return "\\textipa{%s}" % rv
    else:
        return ''.join(rv.replace('\\ae','Q').split(' '))

# returns a list of constraints required for equality to hold
def constrain_phonemes(ps,correct):
    cs = []
    l = ps[0]
    ps = list(ps)[1:]
    correct = correct.split(' ')
    cs.append(l == len(correct))
    
    for c in correct:
        if not (c in ipa2char):
            print c
            print 'Unknown phoneme!'
            print '%s' % correct
    correct = [ z3char[ipa2char[c]] for c in correct ]
    assert(len(correct) < maximum_length+1)
    for j in range(len(correct)):
        cs.append(ps[j] == correct[j])
    return cs

def concatenate(p,q):
    r = morpheme()
    lr = r[0]
    r = list(r)[1:]
    lp = p[0]
    p = list(p)[1:]
    lq = q[0]
    q = list(q)[1:]
    
    # generically, p & q might be constrained to be smaller than the maximum length
    maxP = len(p)
    maxQ = len(q)
    
    constrain(lr == lp+lq)
    
    for j in range(maximum_length):
        if j < maxP:
            constrain(Implies(lp > j,
                              r[j] == p[j]))
        maxI = min([maximum_length-j, maxQ])
        constrain(Implies(lp == j,
                          And(*[ Implies(lq > i, r[i+j] == q[i])
                                 for i in range(maxI) ])))
    return tuple([lr]+r)
                          
                          


def last_one(ps):
    constrain(ps[0] > 0)
    ending = Const(new_symbol(), Phoneme)
    for j in range(1,maximum_length+1):
        constrain(Implies(ps[0] == j,ending == ps[j]))
    return ending


def extract_feature(p, sort, realizations, table):
    return_value = Const(new_symbol(), sort)
    renderings = [str(v) for v in realizations ]
    table = [ (realizations[renderings.index(name)],
               [ z3char[ipa2char[m]] for m in matches.split(' ') ])
              for name, matches in table.iteritems() ]
    if renderings[0] == 'No'+str(sort): # this will be the default case
        expression = realizations[0]
    else:
        expression = table[0][0] # pick a default arbitrarily
        table = table[1:]
    for answer, possibilities in table:
        expression = If(Or(*[ p == possibility for possibility in possibilities ]),
                        answer,
                        expression)
    constrain(return_value == expression)
    return return_value

def voice(p):
    return extract_feature(p, Voicing, voices, voice_table)
def manner(p):
    return extract_feature(p, Manner, manners, manner_table)
def place(p):
    return extract_feature(p, Place, places, place_table)
def sibilant(p):
    return extract_feature(p, Sibilant, sibilance, sibilant_table)


def primitive_string():
    thing = structural(morpheme())
    def evaluate_string(i):
        return thing
    def print_string(m):
        return extract_string(m,thing)
    m = real()
    constrain(m == logarithm(44)*thing[0])
    return evaluate_string, m, print_string

enum_rule('VOICE', list(voices))
enum_rule('PLACE', list(places)[1:])
enum_rule('MANNER', list(manners)[1:])
enum_rule('SIBILANT', list(sibilance)[1:])

rule('VOICE-GUARD', [],
     lambda m: '?',
     lambda i: True)
rule('VOICE-GUARD', ['VOICE'],
     lambda m, g: g,
     lambda i, f: f == voice(i['last']))

rule('MANNER-GUARD', [],
     lambda m: '?',
     lambda i: True)
rule('MANNER-GUARD', ['MANNER'],
     lambda m, g: g,
     lambda i, f: f == manner(i['last']))

rule('PLACE-GUARD', [],
     lambda m: '?',
     lambda i: True)
rule('PLACE-GUARD', ['PLACE'],
     lambda m, g: g,
     lambda i, f: f == place(i['last']))

rule('SIBILANT-GUARD', [],
     lambda m: '?',
     lambda i: True)
rule('SIBILANT-GUARD', ['SIBILANT'],
     lambda m, g: g,
     lambda i, f: f == sibilant(i['last']))

rule('GUARD', ['VOICE-GUARD','MANNER-GUARD','PLACE-GUARD','SIBILANT-GUARD'],
     lambda m, v, ma, g, s: ("[ %s %s %s %s ]" % (v,g,ma,s)).replace(' ?',''),
     lambda i, f, ma, g, s: And(f,ma,g,s))


rule('STEM', [],
     lambda m: 'lemma',
     lambda i: i['lemma'])
rule('MORPHEME',['STRING'],
     lambda m,s: s,
     lambda i,s: s)
rule('MORPHEME',['STEM'],
     lambda m,s: s,
     lambda i,s: s)
rule('RETURN',['STEM','STRING'],
     lambda m, stem, suffix: stem if suffix == '\\textipa{}' else "(append %s %s)" % (stem,suffix),
     lambda i, p, q: concatenate(p,q))
rule('CONDITIONAL',['GUARD','RETURN','CONDITIONAL'],
     lambda m, p,q,r: "(if %s %s %s)" % (p,q,r),
     lambda i, p,q,r: conditional(p,q,r))
rule('CONDITIONAL',['RETURN'],
     lambda m, r: r,
     lambda i, r: r)
primitive_rule('STRING',
               primitive_string)


def train_on_matrix(observations):
    global maximum_length
    maximum_length = max([len(w.split(' ')) for s,t,w in observations ])
    # one program for each tense
    programs = {}
    for (stem,t,inflected) in observations:
        if not (t in programs):
            programs[t] = generator(3,'CONDITIONAL')
    push_solver() # frame for the training data
    # one stem for each form
    stems = {}
    inputs = {}
    for (stem,t,inflected) in observations:
        if not (stem in stems):
            stems[stem] = morpheme()
            inputs[stem] = {'lemma': stems[stem],
                            'last': last_one(stems[stem])}
        cs = constrain_phonemes(programs[t][0](inputs[stem]),inflected)
        constrain(cs)
    flat_stems = [ inputs[i]['lemma'] for i in inputs ]
    stem_length = [logarithm(44)*s[0] for s in flat_stems ]
    program_length = [programs[p][1] for p in programs ]
    description_length = summation(stem_length + program_length)
    
    def printer(m):
        model = ""
        for p in programs:
            model += tense_name[p]
            model += '\t'
            model += programs[p][2](m)
            model += "\n"
        for stem in stems:
            model += "stem[%s] = %s\n" % (stem,extract_string(m,inputs[stem]['lemma']))
        return model
    compressionLoop(printer,description_length)
    return programs


def testing_likelihood(programs):
    global maximum_length
    minimum_maximum = maximum_length
    solver = get_solver()
    likelihood = 0.0
    for test in verbs:
        push_solver()
        maximum_length = max([len(test[t].split(' ')) for t in programs ] + [minimum_maximum])
        test_input = {'lemma': morpheme() }
        test_input['last'] = last_one(test_input['lemma'])
        noise_penalties = []
        exception = {}
        for t in programs:
            inflection_length = len(test[t].split(' '))
            cs = constrain_phonemes(programs[t][0](test_input),test[t])
            exception[t] = boolean()
            constrain(Implies(Not(exception[t]),And(*cs)))
            noise_penalties.append(If(Not(exception[t]),
                                      0.0,
                                      -logarithm(epsilon) + logarithm(44)*inflection_length))
        description_length = summation([test_input['lemma'][0]*logarithm(44)] + noise_penalties)
        model,l = compressionLoop(lambda m: "",description_length,verbose = False,enforce_structure = False)
        assert model != 'FAIL'
        model = get_recent_model()
        likelihood -= l
        # check if we passed the test
        exception = dict([(e,extract_bool(model,exception[e]) == 'True') for e in exception ])
        if not any(exception.values()): print "Passed %s" % test[0]
        else: print "Failed %s" % test[0]
        # stemming
        stem = extract_string(model,test_input['lemma'], tipa = False)
        for t in programs:
            flattened_test = ''.join(test[t].replace('\\ae','Q').split(' '))
            if exception[t]: print flattened_test
            else: 
                if flattened_test == stem: print stem
                else:
                    suffix = flattened_test[len(stem):]
                    print stem,'+',suffix
    print "LIKELIHOOD",likelihood

if __name__ == '__main__':
    N = int(sys.argv[1])
    models = {'sparse': sparse_lexicon,
              'lexicon': sample_corpus,
              'coupled': coupled_sparsity}
    print 'Using %s' % sys.argv[2]
    model = models[sys.argv[2]]
    training = model(N)
    latexTable(training)
    programs = train_on_matrix(training)
    testing_likelihood(programs)
