#!/usr/bin/python

# import collections

# Get list of clusters with a number of words in a cluster.
# Return    cluster2size list 
def getListCluster2Size( cluster_X ):
    "Get list of clusters with a number of words in a cluster"

    cluster_sizes = set()
    cluster2size = [] # from cluster number to size of this cluster
    for i, clu in enumerate (cluster_X): 
        cluster_sizes.add  ( len(clu))
        cluster2size.append( len(clu))
        #if i < 7:                                                   # list of clusters
        #    print("Cluster {0} has {1} words.".format(i, len(clu)))
            ### print("cluster {0} has {1} words. cluster is {2}".format(i, len(clu), clu))

    #print("... cluster_sizes=set(). Sorted clusters sizes are [:7] ")
    #print(sorted(cluster_sizes)[:7])

    return cluster2size, cluster_sizes;
