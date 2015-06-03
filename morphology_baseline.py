import sys
import os
from lexicon import *
import re

if sys.argv[1] == 'baseline':
    os.system('python corpus.py > /tmp/phonetic_lexicon')
    os.system('morfessor  -t /tmp/phonetic_lexicon -S /tmp/segmentation -d ones')
    segmentation_file = '/tmp/segmentation'
else:
    segmentation_file = sys.argv[1]

attempts = 0
correct = 0

def correct_explanation(morphemes,suffix):
    suffix = suffix.replace('@','').replace(' ','')
    suffix = re.sub(r'-.?','',suffix)
    print suffix
    if len(morphemes) == 1:
        return suffix in ['IRR','']
    if len(morphemes) == 2:
        if morphemes[1] == '@d': return suffix == '+d'
        if morphemes[1] == 't': return suffix == '+d' or suffix == '+t'
        if morphemes[1] == 'd': return suffix == '+d'
        if morphemes[1] == '@z': return suffix == '+s'
        if morphemes[1] == 's': return suffix == '+s'
        if morphemes[1] == 'z': return suffix == '+s'
        if morphemes[1] == 'IN': return suffix == '+ing'
        if morphemes[1] == 'n': return suffix == '+n'
        if morphemes[1] == '@n': return suffix == '+n'
        return False
    return False

with open(segmentation_file) as f:
    for l in f:
        if l[0] == '#':
            continue
        l = l.strip()
        # remove frequency
        l = l.split(' ')
        if l[0].isdigit():
            l = ' '.join(l[1:])
        else:
            l = ' '.join(l)
        
        attempts += 1
        
        # failure: more than 2 morphemes
        morphemes = l.split(' + ')
        if len(morphemes) > 2: continue
        
        
        inflection = ' '.join([c for c in ''.join(morphemes) ]).replace('Q','\\ae')

        for v in range(len(celex_inflections)):
            candidate_explanations = []
            for i in range(5):
                if celex_inflections[v][i] == inflection:
                       candidate_explanations.append(celex_stem[v][i][1])
#            if len(candidate_explanations) == 0:
#                print morphemes,inflection
            if any([correct_explanation(morphemes,suffix) 
                    for suffix in candidate_explanations ]):
                correct += 1
                break
#            else:
#                print morphemes,candidate_explanations

print float(correct)/attempts
