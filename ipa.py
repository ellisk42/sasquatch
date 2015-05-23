#!/usr/bin/env python
# author: eyal dechter 

import sys
import fileinput

import urllib
import urllib2
from bs4 import BeautifulSoup

phoneme_code = {'AO': 'a', 'AA': 'a', 'IY': 'i', 'UW': 'u', 'EH': 'E', 'IH': 'I',
                'UH': 'U', 'AH': todo}

def load_pronunciations():
    p = {}
    with open('cmudict-0.7b','r') as f:
        for l in f:
            if len(l) == 0 or l[0] == ';':
                continue
            if '(' in l or ')' in l: continue
            pieces = [ x for x in l.strip().split(' ') if len(x) > 0 ]
            p[pieces[0]] = ' '.join([ phoneme_code[x] for x in pieces[1:] ])
            return p

URL= "http://upodn.com/phon.asp"

def ipa(intext):
    """ Query upondn.com for IPA form of input word or words. 
    <word> should be a string or list of strings. 
    Returns a dictionary mapping input words to IPA strings. 
    """
    if isinstance(intext, list):
        intext = (' ').join(intext)
    
    form_data = { 'ipa' : 0, 
                  'intext' : intext}
    params = urllib.urlencode(form_data)
    response = urllib2.urlopen(URL, params)
    
    data = response.read()
    
    soup = BeautifulSoup(data)

    table = soup.find_all('table')[0]

    incell = table.find_all('td')[0].text.split()

    outcell = table.find_all('td')[1].text.split()

    out = dict(zip(incell, outcell))
    
    return out

if __name__ == "__main__":
    load_pronunciations()
    if len(sys.argv) == 1 or sys.argv[1] in ["-h", "--help"]: 
        print """ ipa.py usage: `ipa.py word1 word2 ...` returns csv 
        column 1 words in english 
        column 2 words in ipa """
        exit()
    intext = sys.argv[1:]
    out = ipa(intext)
    print ('\n').join([a + ", " + b for (a, b) in out.items()])
    
