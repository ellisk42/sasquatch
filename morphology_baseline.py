import os
from corpus import *
import celex


lexicon = celex.load_celex()

os.system('python corpus.py > /tmp/phonetic_lexicon')
os.system('morfessor  -t /tmp/phonetic_lexicon -S /tmp/segmentation')


attempts = 0
correct = 0

def correct_explanation(morphemes,suffix,transform):
    if len(morphemes) == 1:
        return suffix == 'null'
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

with open('/tmp/segmentation') as f:
    for l in f:
        if l[0] == '#':
            continue
        l = l.strip()
        # remove frequency
        l = ' '.join(l.split(' ')[1:])
        
        attempts += 1
        
        # failure: more than 2 morphemes
        morphemes = l.split(' + ')
        if len(morphemes) > 2: continue
        
        
        inflection = ' '.join([c for c in ''.join(morphemes) ]).replace('Q','\\ae')

        for v in range(len(verbs)):
            lexical_item = lexical_items[v] # what word are we looking at
            candidate_explanations = []
            for i in range(6):
                if verbs[v][i] == inflection:
                    inflection_code = ['VB','VBD','VBG','VBN','VBP','VBZ'][i]
                    lexical_inflection,suffix,transform = lexicon[lexical_item][inflection_code]
                    candidate_explanations.append((suffix,transform))
#            if len(candidate_explanations) > 0:
#                print morphemes,candidate_explanations
            if any([correct_explanation(morphemes,suffix,transform) 
                    for suffix,transform in candidate_explanations ]):
                correct += 1
                break
#            else:
#                print morphemes,candidate_explanations

print float(correct)/attempts
