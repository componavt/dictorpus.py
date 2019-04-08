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

from sklearn.cluster import KMeans
# from sklearn import cluster
from sklearn import metrics

import configus
model = gensim.models.Word2Vec.load(configus.MODEL_PATH)

n_words = len(model.wv.vocab)
print ("\nNumber of words in vocabulary is {}.".format( n_words ))
for key, value in model.wv.vocab.items(): # print all words from vocabulary
    print( key, value)
    break

# dictionaries: from word to index, from index to word
word_idx = dict((word, i) for i, word in enumerate(model.wv.vocab))
idx_word = dict((i, word) for i, word in enumerate(model.wv.vocab))

for word, idx in word_idx.items():
    # word = 'tütär'
    if word in model:
        print("word_idx[{0}]={1}".format(word, idx ))
        print("idx_word[{1}]={0}\n".format(word, idx ))
    break

X = model[ model.wv.vocab ]

# todo: How to select number of clusters?..
# answer: example with graph, where axis-X is number of clusters

#Error: no centroid defined for empty cluster.
#Try setting argument 'avoid_empty_clusters' to True

NUM_CLUSTERS=200

### NLTK ########################
###kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=1) #repeats=25
###cluster_arrays = kclusterer.cluster(X, assign_clusters=True, trace=True)
###print("cluster_arrays: found.")
# print("cluster_arrays: {0}".format(cluster_arrays))

###words = list( model.wv.vocab )
###for i, word in enumerate(words):  
###    print (word + ":" + str(cluster_arrays[i]))
###    if i == 7:
###        break       # too huge list 'words'

### plt.scatter(Y[:, 0], Y[:, 1], c=cluster_arrays, s=3,alpha=.5)
 
#for j in range(len(words)):
#   plt.annotate(cluster_arrays[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#   print ("%s %s" % (cluster_arrays[j],  words[j]))

kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_jobs=4)
kmeans.fit(X)
  
labels    = kmeans.labels_
centroids = kmeans.cluster_centers_
  
print ("\nCluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
  
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
  
silhouette_score = metrics.silhouette_score(X, labels, metric='cosine')
  
print ("Silhouette_score: ")
print (silhouette_score)

# how to calculate number of points in each kluster?
n_clusters_result = len(set(labels)) - (1 if -1 else 0)
print("Number of clusters: {0}".format( n_clusters_result))
# print("labels[0, 10]: {0}".format( labels[0:10]))
#        labels[0, 10]: [ 51 131 119 152  87 167  80 147   4  97]

# gather words to created cluster
cluster_arrays = [X[labels == i] for i in range(n_clusters_result)]

for i, clu in enumerate (cluster_arrays): 
    print("Cluster {0} has {1} words.".format(i, len(clu)))
    # print("cluster {0} has {1} words. cluster is {2}".format(i, len(clu), clu))
print

# todo draw words:
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

#Plot the clusters obtained using k means
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
 
model_tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model_tsne.fit_transform(X)
 
plt.scatter(Y[:, 0], Y[:, 1], c=kmeans.labels_, s=3,alpha=.5, cmap='rainbow') 
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black') 

# todo (first, move centroids via tsne.transform...) 
# Plot the centroids as a white X
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='black', zorder=10)

 
#for j in range(len(words)):
#   plt.annotate(cluster_arrays[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#   print ("%s %s" % (cluster_arrays[j],  words[j]))
 
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
#                     c=kmeans[0],s=50)
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
