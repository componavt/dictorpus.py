#!/bin/bash
for i in `seq 4 3000`;
do
   echo `python3 clustering_kmeans.py $i`
done

# generate video
# ffmpeg -framerate 10 -pattern_type glob -i 'min_count_4_window_2_k-means_*-clusters.png' -c:v libx264 -pix_fmt yuv420p out411.mp4
# 
# scale to 1280x1024
# ffmpeg -framerate 30 -pattern_type glob -i 'min_count_4_window_2_k-means_*-clusters.png' -s 1280x1024 -c:v libx264 -pix_fmt yuv420p out844_fps_30.mp4
