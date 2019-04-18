#!/bin/bash
for i in `seq 1 3000`;
do
    let eps=$((1 / i))
    eps=$(echo "1/$i" | bc -l)

    echo "eps=$eps"

    echo "python3 clustering_dbscan.py $eps 2"
#   echo `python3 clustering_dbscan.py 1/$i 2`
                                           # 2 is MIN_SAMPLES (number of neighbours)
done

# generate video
# ffmpeg -framerate 10 -pattern_type glob -i 'min_count_4_window_2_k-means_*-clusters.png' -c:v libx264 -pix_fmt yuv420p out411.mp4
# 
# scale to 1280x1024
# ffmpeg -framerate 30 -pattern_type glob -i 'min_count_4_window_2_k-means_*-clusters.png' -s 1280x1024 -c:v libx264 -pix_fmt yuv420p out844_fps_30.mp4
