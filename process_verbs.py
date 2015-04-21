import ipa

def tipa(web):
    # These conversions are enough to cover the first 1000 stems (and their inflections)
    # Going beyond these might necessitate adding more conversions from Unicode to TIPA
    conversions = [(u"\u026a",'I'),
                   (u"\xf0",'D'),
                   (u"\u014b",'N'),
                   (u"\xe6",'\\ae'),
                   (u"\u0259",'@'),
                   (u"\u0252",'5'),
                   (u"\u03b8",'T'),
                   (u"\u025b",'E'),
                   (u"\u028a",'U'),
                   (u"\u0251",'a'),
                   (u"\u0283",'S'),
                   (u"\u0254",'0'),
                   (u"\u0292",'Z')]
    result = []
    for w in web:
        for o,n in conversions:
            if o == w:
                w = n
        result.append(w)
    return ' '.join(result)


scores = []
conjugations = {}
lemma_popularity = {}
with open('verbs','r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        [frequency,tense,form,meaning,otherFrequency,transform,suffix] = line.split(',')
        if '-' in meaning or '-' in form:
            continue
        lemma_popularity[meaning] = int(frequency) + lemma_popularity.get(meaning,0)
        conjugations[meaning] = conjugations.get(meaning,{})
        conjugations[meaning][tense] = form
        scores.append((-int(otherFrequency),form))

lemma_popularity = sorted([ (-lemma_popularity[l],l) for l in lemma_popularity.keys() ])
N = 100
stems = [ s[1] for s in lemma_popularity[:N] ]

for stem in stems:
    if 6 != len(conjugations[stem]):
        continue
    vs = sorted(conjugations[stem].keys())
    inflections = [ conjugations[stem][v] for v in vs ]
    phonetics = ipa.ipa(inflections)
    print '[' + ','.join([ '"' + tipa(phonetics[inflection]).encode('unicode-escape') + '"'
                           for inflection in inflections ]) + '],'
    


