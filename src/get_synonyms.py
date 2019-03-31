#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Gets synonyms for some word.

import logging
import codecs
import operator
import collections

import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim

import lib.filter_vocab_words
import lib.string_util
#import lib.synset

import configus
model = gensim.models.Word2Vec.load(configus.MODEL_PATH)

source_words = [u'madal', u'ald', u'alembaine', u'lainiž', u'da', u'dai', u'i', u'ende', u'edel', u'agj', u'lop', u'ajada', u'tüpäkzuda', u'armaz', u'žalanni', u'brusak', u'torokan', u'habaine', u'habasine', u'havad', u'šaug', u'hond', u'huba', u'hökkähtuda', u'ladidas', u'kangaz', u'paltin', u'keskes', u'kesknezoi', u'kipätk', u'kipätkoine', u'kirj', u'knig', u'kittas', u'ülendeldas', u'klub', u'sebr', u'koht', u'vac', u'korm', u'tuk', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'a', ]

print ("Number of words in vocabulary is {}.".format( len(model.wv.vocab)))

#words = lib.filter_vocab_words.filterVocabWords( source_words, model.wv.vocab )
words = lib.filter_vocab_words.filterVocabWords( source_words, model.wv.vocab )
print ("After filter")
print (lib.string_util.joinUtf8( ",", words ))  # after filter, now there are only words with vectors

print("model.wv.most_similar('madal'): {0}".format(model.wv.most_similar("madal")))
#print(model.wv.most_similar("madal"))

#while words:
#    #print string_util.joinUtf8( ",", words )
#    out_word = model.doesnt_match(words)
#    print u"    - '{}'".format( out_word )
#    words.remove( out_word )
