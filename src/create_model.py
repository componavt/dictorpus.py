#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Prints list of synonyms in synset. -1 the most outer word, ... while |synset|>0

import logging
import codecs
import operator
import collections

import os, sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim

dirname  = './data'
fname = 'vepkar-2019-03-30-vep.txt'

sentences = []
with open(os.path.join(dirname, fname), encoding="utf8") as f:
    for line in f:
        tokens = line.rstrip('\n').lower().split('|')
        sentences.append( tokens )
        #print (tokens)

model = gensim.models.Word2Vec(sentences, min_count=5)
model.save('./model/min_count5.model')


# 2/6 = |IntS|/|S|, [[сосредоточиваться]],  IntS(сосредоточиваться сосредотачиваться)  OutS(собираться отвлекаться фокусироваться концентрироваться) 
# source_words = [u'сосредоточиваться', u'сосредотачиваться', u'собираться', u'отвлекаться', u'фокусироваться', u'концентрироваться']
source_words = [u'лить', u'кутить', u'сосредоточиваться', u'сосредотачиваться', u'собираться', u'отвлекаться', u'фокусироваться', u'концентрироваться']

# 0/6 = |IntS|/|S|, [[абсолют]],  OutS(абсолют логос первооснова творец совершенство идеал) 

#words = lib.filter_vocab_words.filterVocabWords( source_words, model.wv.vocab )
#print string_util.joinUtf8( ",", words )                                # after filter, now there are only words with vectors

#while words:
    #print string_util.joinUtf8( ",", words )
#    out_word = model.doesnt_match(words)
#    print u"    - '{}'".format( out_word )
#    words.remove( out_word )
