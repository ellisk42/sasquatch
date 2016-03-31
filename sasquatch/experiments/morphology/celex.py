

def load_celex():
    lexicon = {}
    with open('verbs') as f:
        for l in f:
            if 'CELEX' in l: continue
            parts = l.split(',')
            tense = parts[1]
            inflection = parts[2]
            stem = parts[3]
            stem_transform = parts[5]
            suffix = parts[6].strip()
            
            lexicon[stem] = lexicon.get(stem,{})
            lexicon[stem][tense] = (inflection,suffix,stem_transform)
            
    return lexicon
#print load_celex()
