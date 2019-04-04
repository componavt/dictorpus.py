#!/usr/bin/env python
# -*- coding: utf-8 -*-

# K Means Clustering of words with NLTK Library.
# see http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/

import logging
import codecs
import operator
import collections

import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim

import lib.filter_vocab_words
import lib.string_util

from nltk.cluster import KMeansClusterer
import nltk

import configus
model = gensim.models.Word2Vec.load(configus.MODEL_PATH)

n_words = len(model.wv.vocab)
print ("\nNumber of words in vocabulary is {}.".format( n_words ))

#X = model.wv.vocab[n_words]
#X = model.wv.vocab
# X = model[n_words]      # TypeError: 'int' object is not iterable
X = model[ model.wv.vocab ]

NUM_CLUSTERS=33
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=5) #repeats=25
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)

words = list( model.wv.vocab )
for i, word in enumerate(words):  
    print (word + ":" + str(assigned_clusters[i]))

#words = lib.filter_vocab_words.filterVocabWords( source_words, model.wv.vocab )
#words = lib.filter_vocab_words.filterVocabWords( source_words, model.wv.vocab )
#print ("After filter")
#print (lib.string_util.joinUtf8( ",", words ))  # after filter, now there are only words with vectors

#X = model[model.vocab]

#print("model.wv.most_similar('madal'): {0}".format(model.wv.most_similar("madal")))
#print(model.wv.most_similar("madal"))

#while words:
#    #print string_util.joinUtf8( ",", words )
#    out_word = model.doesnt_match(words)
#    print u"    - '{}'".format( out_word )
#    words.remove( out_word )
