#!/usr/bin/python

import os, sys

# Read file with lemmas and part of speech (POS), return dictionary word to POS.
#
# Format:  lemma|POS joined by pipe.
# Example: Karjala|PROPN
def readLemmaPOSFile( filename ):
    "Skip words which are absent in the model vocabulary"
    
    # result filtered list
    result = dict()
   
    result["Karjala"] = "PROPN"

    dirname  = './data'
    fname = 'vepkar-2019-04-05-lemma-pos-vep.txt'

    lines = []
    with open(os.path.join(dirname, fname), encoding="utf8") as f:
        for line in f:
            (lemma, POS) = line.rstrip('\n').lower().split('|')
            result[lemma] = POS

    # test
    #word = "Karjala"
    #POS = result.get(word)
    #print ("POS('{0}') = {1}".format(word, POS))

    return result

