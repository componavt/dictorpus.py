#!/usr/bin/env python
# -*- coding: utf-8 -*-

# K Means Clustering of words with scikit-learn library.
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
import lib.cluster2size
import lib.read_lemma_pos_file

import numpy as np 

from sklearn.cluster import KMeans
from sklearn import metrics

import configus
model = gensim.models.Word2Vec.load(configus.MODEL_PATH)

n_words = len(model.wv.vocab)
print ("\nNumber of words in vocabulary is {}.".format( n_words ))
for key, value in model.wv.vocab.items(): # print fist word from vocabulary
    print( key, value)
    break

# dictionaries: from word to index, from index to word
print ("\nword_idx - from word to index, idx_word - from index to word")
word_idx = dict((word, i) for i, word in enumerate(model.wv.vocab))
idx_word = dict((i, word) for i, word in enumerate(model.wv.vocab))

for word, idx in word_idx.items():
    # word = 'tütär'
    if word in model.wv:
        print("word_idx[{0}]={1}".format(word, idx ))
        print("idx_word[{1}]={0}".format(word, idx ))
    break

X = model.wv [ model.wv.vocab ]

# todo: How to select right number of clusters?..
# Answer: draw histogram: sizes of clusters depends on number of clusters (axis-X)

#Error: no centroid defined for empty cluster.
#Try setting argument 'avoid_empty_clusters' to True

#NUM_CLUSTERS=200
NUM_CLUSTERS = int(float(sys.argv[1]))
print("The first argument (NUM_CLUSTERS) is: {}.".format(NUM_CLUSTERS))

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
#print ("Centroids data")
#print (centroids) # too huge numbers

score = kmeans.score(X)
silhouette_score = metrics.silhouette_score(X, labels, metric='cosine')
print ("Score (Sum of distances of samples to their closest cluster center): {}".format(score))
print ("Silhouette_score: {}".format(silhouette_score))

# how to calculate number of points in each cluster?
n_clusters_result = len(set(labels))
print("Number of clusters: {0}".format( n_clusters_result))
print("Number of labels: {0}".format( len(labels)))
# print("labels[0, 10]: {0}".format( labels[0:10]))
#        labels[0, 10]: [ 51 131 119 152  87 167  80 147   4  97]

# gather words to created cluster
cluster_X     = [X       [labels == i] for i in range(n_clusters_result)]
#cluster_words = [idx_word[labels == i] for i in range(n_clusters_result)]

lemma_POS = lib.read_lemma_pos_file.readLemmaPOSFile(configus.LEMMA_POS_PATH)

# get sets of words for each cluster
word_sets = [set() for _ in range(NUM_CLUSTERS)]

# gather POSes for each cluster
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

SMALL_CLUSTER_SIZE = 10 # calculates POS number only for small clusters

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
            print("{0}. POS='{1}', {2:.1f}%, {3} POS of {4} words.".format(i, pos_, pos_percent, n_pos, 
                        len(word_sets[i]))) # number of words in cluster

print("Number of small clusters is {0} out of {1} clusters.".format( n_small_clusters, NUM_CLUSTERS))

# todo draw words:
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

#Plot the clusters obtained using k means
import matplotlib.pyplot as plt
import matplotlib.cm     as mpl_cm          # for Brewer etc colourmaps
import matplotlib.colors as colors
from sklearn.manifold import TSNE


# continuous colormap
cpal = 'PuOr_r'
cmap_cont = mpl_cm.get_cmap(cpal)

# Add values for bad/over/under:
badcol   = 'grey'
overcol  = 'magenta'
undercol = 'cyan'

cmap_cont.set_bad(  color=badcol,   alpha=0.5 )
cmap_cont.set_over( color=overcol,  alpha=0.5 )
cmap_cont.set_under(color=undercol, alpha=0.5 )


# cluster2size  - list from cluster number to size of this cluster
# cluster_sizes - set of clusters sizes
cluster2size, cluster_sizes = lib.cluster2size.getListCluster2Size( cluster_X )

norm = colors.BoundaryNorm( sorted(cluster_sizes), ncolors=cmap_cont.N, clip=True)

print("\n cluster2size=list[]. Clusters sizes for each cluster are [:7] ")
print(cluster2size[:7])

# Create labels2cluster_size - from word to cluster size
# Given 1) labels - from word to cluster number
#       2) cluster2size - from cluster number to cluster size
labels2cluster_size = []
for i, cluster_idx in enumerate (labels):
    size_ = cluster2size[ cluster_idx ]
    # print("i={0}, cluster_idx={1}, size_={2}".format(i, cluster_idx, size_))
    labels2cluster_size.append( size_ )

print("\n labels2cluster_size=... Clusters sizes for each word [:7] ")
print(labels2cluster_size[:7])


plt.figure(figsize=(8.5, 6.4), dpi=100) # = 640x480

ax1 = plt.subplot(2, 1, 1)
plt.title("POS (one type) depends on cluster's size");

pos_color = {
  "noun": "red",
  "verb": "green",
  "adv": "blue",
  "adj": "yellow"
}
xy = dict()
for i, (pos_, pos_percent, n_pos) in cluster_POS_quality.items():
    x = n_pos
    y = pos_percent
    xy[ (x,y) ] = 1 + xy.get((x,y), 0)  # 0 is default_value, if we increment (x,y) in first time
                                        #                     and xy.get((x,y)) do not exist
for i, (pos_, pos_percent, n_pos) in cluster_POS_quality.items():
    x = n_pos
    y = pos_percent
    color_ = "grey"
    if pos_ in pos_color:
        color_ = pos_color[pos_]
    size_ = xy[ (x, y) ]
    plt.scatter(x, y, c=color_, s=size_*200,alpha=.1, cmap='set3', zorder=10)
    plt.annotate(size_, (x, y))

ax1.set_xticks([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], minor=False)
ax1.set_yticks([20, 40, 50, 60, 80, 100], minor=False)
# ax1.axhline(50, linestyle='--', color='red') # horizontal red line - 50%
plt.hlines(y=50, xmin=1, xmax=10, linestyle='--', color='red', zorder=5) # horizontal red line - 50%
plt.ylim([20,110])
plt.xlim([1,10])

##fig = plt.figure()
plt.ylabel('POS one type in cluster, %', color='green') #, fontsize=15)
plt.text(0.95, 0.97, 
        "Clusters: {0} (small) of {1}. Small cluster has 2-{2} elements.\nScore = {3: 10.2f}     Silhouette ={4: 8.3f} ".format(
            n_small_clusters, NUM_CLUSTERS, SMALL_CLUSTER_SIZE, score, silhouette_score),
        verticalalignment='top', horizontalalignment='right', transform = ax1.transAxes)

plt.text(0.95, 0.01, 'Words in cluster',
        verticalalignment='bottom', horizontalalignment='right',
        transform = ax1.transAxes,
        color='green') # , fontsize=15)

plt.subplot(2, 1, 2 )
plt.xlabel("word2vec model={0}. k-means. {1} clusters".format(
    configus.MODEL_NAME, NUM_CLUSTERS))
#plt.title('word2vec model={0}. k-means. {1} clusters'.format(
#    configus.MODEL_NAME, NUM_CLUSTERS), y=-0.01);

model_tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y = model_tsne.fit_transform(X)
#centers_new = model_tsne.fit_transform(kmeans.cluster_centers_)

#plt.scatter(Y[:, 0], Y[:, 1], c=kmeans.labels_, s=3,alpha=.5, cmap='rainbow') 
plt.scatter(Y[:, 0], Y[:, 1], c=labels2cluster_size, s=3,alpha=.5, cmap = cmap_cont, norm=norm) 

#plt.scatter(centers_new[:,0], centers_new[:,1], color='black') 

# todo (first, move centroids via tsne.transform...) 
# Plot the centroids as a white X
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='black', zorder=10)

filename = '{0}_k-means_{1:04d}-clusters'.format(configus.MODEL_NAME, NUM_CLUSTERS)
plt.savefig(configus.SRC_PATH + "data/fig/kmeans/" + filename, bbox_inches='tight', dpi=900)
# plt.rcParams["figure.figsize"] = fig_size

# plt.show()

# save to file parameters and quality results
filename_quality = '{0}_k-means'.format( configus.MODEL_NAME )

{1:04d}-clusters
NUM_CLUSTERS
