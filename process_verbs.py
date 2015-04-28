import ipa

def tipa(web):
    # These conversions are enough to cover the first 1000 stems (and their inflections)
    # Going beyond these might necessitate adding more conversions from Unicode to TIPA
    conversions = [(u"\u026a",'I'),
                   (u"\xf0",'D'),
                   (u"\u014b",'N'),
                   (u"\xe6",'\\ae'),
                   (u"\u02a7","t S"),
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

# celex uses the British version of the word practice;
# but this doesn't work with our IPA converter
if False:
    practice = ["p r a c t i s e","p r a c t i s e d","p r a c t i s i n g","p r a c t i s e d","p r a c t i s e","p r a c t i s e s"]
    practice = [ p.replace(' ','').replace('practis','practic') for p in practice ]
    phonetics = ipa.ipa(practice)
    print '[' + ','.join([ '"' + tipa(phonetics[inflection]).encode('unicode-escape') + '"'
                           for inflection in practice ]) + '],'
    os.exit()
if True:
    practice = ["a n a l y s e","a n a l y s e d","a n a l y s i n g","a n a l y s e d","a n a l y s e"]
    practice = [ p.replace(' ','').replace('analys','analyz') for p in practice ]
    phonetics = ipa.ipa(practice)
    print '[' + ','.join([ '"' + tipa(phonetics[inflection]).encode('unicode-escape') + '"'
                           for inflection in practice ]) + '],'
    os.exit()

scores = []
conjugations = {}
lemma_popularity = {}
irregular = {}
with open('verbs','r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        [frequency,tense,form,meaning,otherFrequency,transform,suffix] = line.split(',')
        if '-' in meaning or '-' in form:
            continue
        if transform != "null":
            irregular[meaning] = True
        lemma_popularity[meaning] = int(frequency) + lemma_popularity.get(meaning,0)
        conjugations[meaning] = conjugations.get(meaning,{})
        conjugations[meaning][tense] = form
        scores.append((-int(otherFrequency),form))
# removed verbs without all 6 inflections
bad_stems = []
for stem in conjugations:
    if 6 != len(conjugations[stem]):
        bad_stems.append(stem)
for stem in bad_stems:
    del lemma_popularity[stem]
    del conjugations[stem]

lemma_popularity = sorted([ (-lemma_popularity[l],l) for l in lemma_popularity.keys() ])
N = 0
stems = [ s[1] for s in lemma_popularity[:N] ]

for stem in stems:
    vs = sorted(conjugations[stem].keys())
    inflections = [ conjugations[stem][v] for v in vs ]
    phonetics = ipa.ipa(inflections)
    print '[' + ','.join([ '"' + tipa(phonetics[inflection]).encode('unicode-escape') + '"'
                           for inflection in inflections ]) + '],'

print ','.join([ "'" + tipa(ipa.ipa(i)[i]).encode('unicode-escape') + "'" for i in irregular ])
    
