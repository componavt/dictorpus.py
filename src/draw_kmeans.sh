#!/bin/bash
for i in `seq 4 3000`;
do
   echo `python3 clustering_kmeans.py $i`
done
