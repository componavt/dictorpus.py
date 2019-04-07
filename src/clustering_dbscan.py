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
import numpy as np 

from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import DBSCAN

import matplotlib.colors as colors
import matplotlib.cm as cmx

import configus
model = gensim.models.Word2Vec.load(configus.MODEL_PATH)

n_words = len(model.wv.vocab)
print ("\nNumber of words in vocabulary is {}.".format( n_words ))

X = model[ model.wv.vocab ]

# eps is radius, the minimum distance between two points
# min_samples - number of data points in a neighborhood
dbscan = DBSCAN(metric='cosine', eps=0.02, min_samples=2, n_jobs=4)
clusters = dbscan.fit_predict(X) # X is matrix, where each row corresponds to one word-vector

outliers = X[clusters == -1]
print("outliers number: {0}".format( len (outliers)))

n_clusters = len(set(clusters)) - (1 if -1 else 0)
print("clusters number: {0}".format( n_clusters))

cluster_arrays = [X[clusters == i] for i in range(n_clusters)]

for i, clu in enumerate (cluster_arrays): 
    print("cluster {0} has {1} words.".format(i, len(clu)))
print

words = list( model.wv.vocab )
for i, word in enumerate(words):  
    print (word + ":" + str(clusters[i]))
    if i == 3:
        break       # too huge list 'words'

# Plot the clusters
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
 
model_tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model_tsne.fit_transform(X)

my_cmap = plt.get_cmap('Paired') 
#cNorm  = colors.Normalize(vmin=0, vmax=n_clusters-1)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_cmap)
#print("scalarMap.get_clim(): {0}".format(scalarMap.get_clim()))

fig, ax = plt.subplots()
for j in range(len(clusters)):
    color = my_cmap(clusters[j])
    transparence = 0.7
    size = 30
    if -1 == clusters[j]:   # let's outliers will be almost invisible
        color = 'silver'
        transparence = 0.3
        size = 4 
    ax.scatter(Y[j, 0], Y[j, 1], c=color, s=size, #label=color,
               alpha=transparence, edgecolors='none')
#   plt.annotate(clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')

# s - size in points
# alpha - transparency 
#plt.scatter(Y[:, 0], Y[:, 1], c=clusters, s=1,alpha=.3)
##plt.scatter(Y[:, 0], Y[:, 1], c=clusters)
 
#for j in range(len(words)):
#   plt.annotate(clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#   print ("%s %s" % (clusters[j],  words[j]))
 
plt.show()

sys.exit("Stop and think.")

#ax.set_title('K-Means Clustering')
#ax.set_xlabel('GDP per Capita')
#ax.set_ylabel('Corruption')
#plt.colorbar(scatter)


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
