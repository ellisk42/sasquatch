#!/usr/bin/env python
# author: eyal dechter 

import sys
import fileinput

import urllib
import urllib2
from bs4 import BeautifulSoup

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
    if len(sys.argv) == 1 or sys.argv[1] in ["-h", "--help"]: 
        print """ ipa.py usage: `ipa.py word1 word2 ...` returns csv 
        column 1 words in english 
        column 2 words in ipa """
        exit()
    intext = sys.argv[1:]
    out = ipa(intext)
    print ('\n').join([a + ", " + b for (a, b) in out.items()])
    
