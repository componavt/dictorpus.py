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
import lib.part_of_speech
import lib.read_lemma_pos_file

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
print ("\nword_idx - from word to index, idx_word - from index to word")
word_idx = dict((word, i) for i, word in enumerate(model.wv.vocab))
idx_word = dict((i, word) for i, word in enumerate(model.wv.vocab))

for word, idx in word_idx.items():
    # word = 'tütär'
    if word in model:
        print("word_idx[{0}]={1}".format(word, idx ))
        print("idx_word[{1}]={0}".format(word, idx ))
    break

X = model[ model.wv.vocab ]

# todo: How to select right number of clusters?..
# Answer: draw histogram: sizes of clusters depends on number of clusters (axis-X)

#Error: no centroid defined for empty cluster.
#Try setting argument 'avoid_empty_clusters' to True

NUM_CLUSTERS=200

kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_jobs=4)
kmeans.fit(X)
  
labels    = kmeans.labels_
centroids = kmeans.cluster_centers_

print ("\n7 pairs: word - cluster number")
words = list( model.wv.vocab )
for i, word in enumerate(words):  
    print (word + ":" + str(labels[i]))
    if i == 7:
        break       # too huge list 'words'

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
print("Number of labels: {0}".format( len(labels)))
# print("labels[0, 10]: {0}".format( labels[0:10]))
#        labels[0, 10]: [ 51 131 119 152  87 167  80 147   4  97]

# gather words to created cluster
cluster_X     = [X       [labels == i] for i in range(n_clusters_result)]
#cluster_words = [idx_word[labels == i] for i in range(n_clusters_result)]

# list of clusters with a number of words in a cluster
for i, clu in enumerate (cluster_X): 
    print("Cluster {0} has {1} words.".format(i, len(clu)))
    # print("cluster {0} has {1} words. cluster is {2}".format(i, len(clu), clu))
    if i == 7:
        break       # too huge list of clusters
print("...\n")

lemma_POS = lib.read_lemma_pos_file.readLemmaPOSFile(configus.LEMMA_POS_PATH)

# get sets of words for each cluster
word_sets = [set() for _ in range(NUM_CLUSTERS)]

# dictionary with lists: cluster_POS[i]={NOUN, VERB, NOUN...} - POSes for i-th cluster
cluster_POS = dict()
for i in range(NUM_CLUSTERS):
    cluster_POS[i] = list()

for i, cluster_idx in enumerate (labels):
    word_ = idx_word[i]
    word_sets[cluster_idx].add(word_)   # save word_ to the cluster idx
    if word_ in lemma_POS:
        pos_ = lemma_POS[word_]
        cluster_POS[cluster_idx].append(pos_)

SMALL_CLUSTER_SIZE = 10

# calculate POS quality for each cluster (all words with the same type of POS means better quality)
cluster_POS_quality = dict()

for i in range(NUM_CLUSTERS):
    pos_ = cluster_POS[i] # list
    if 1 < len(pos_) and len(pos_) < SMALL_CLUSTER_SIZE:
        pos_, pos_percent, n_pos = lib.part_of_speech.getMostFrequentPOS( pos_ )
        cluster_POS_quality[ i ] = (pos_, pos_percent, n_pos)
        # print("{0}. POS='{1}', {2:.1f}%, total {3} POS.".format(i, pos_, pos_percent, n_pos))

# print small clusters
n_small_clusters = 0
print("\nPrint small clusters, where size < {0}".format( SMALL_CLUSTER_SIZE))
for i, set_ in enumerate (word_sets): 
    if 1 < len(set_) and len(set_) < SMALL_CLUSTER_SIZE:
        print("\nword_sets[{0}]  : {1}".format( i, word_sets[i]))
        print("cluster_POS[{0}]: {1}".format( i, cluster_POS[i]))
        n_small_clusters += 1
        if i in cluster_POS_quality:
            (pos_, pos_percent, n_pos) = cluster_POS_quality[i]
            print("{0}. POS='{1}', {2:.1f}%, total {3} POS.".format(i, pos_, pos_percent, n_pos))

print("Number of small clusters is {0} out of {1} clusters.".format( n_small_clusters, NUM_CLUSTERS))

# todo draw words:
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

#Plot the clusters obtained using k means
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.subplot(2, 1, 1)
plt.title("POS (one type) depends on cluster's size. {0} of {1} small clusters (2-{2} elements) .".format(n_small_clusters, NUM_CLUSTERS, SMALL_CLUSTER_SIZE));

pos_color = {
  "noun": "red",
  "verb": "green",
  "adv": "blue",
  "adj": "brown"
}
rng = np.random.RandomState(0)
for i, (pos_, pos_percent, n_pos) in cluster_POS_quality.items():
    
    rand_x = (rng.randn(100) - 50) / 10
    rand_y = (rng.randn(100) - 50)

    color_ = "grey"
    if pos_ in pos_color:
        color_ = pos_color[pos_]
    plt.scatter(abs(n_pos+rand_x), abs(pos_percent+rand_y), c=color_, s=150,alpha=.1, cmap='rainbow') 


plt.subplot(2, 1, 2)
plt.title('word2vec model={0}. k-means. Clusters={1}'.format(configus.MODEL_NAME, NUM_CLUSTERS));

model_tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y = model_tsne.fit_transform(X)
#centers_new = model_tsne.fit_transform(kmeans.cluster_centers_)
 
plt.scatter(Y[:, 0], Y[:, 1], c=kmeans.labels_, s=3,alpha=.5, cmap='rainbow') 
#plt.scatter(centers_new[:,0], centers_new[:,1], color='black') 

# todo (first, move centroids via tsne.transform...) 
# Plot the centroids as a white X
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='black', zorder=10)

 
#for j in range(len(words)):
#   plt.annotate(cluster_X[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#   print ("%s %s" % (cluster_X[j],  words[j]))

#fig, ax = plt.subplots()
#plt.legend()

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
