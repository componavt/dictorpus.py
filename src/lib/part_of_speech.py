#!/usr/bin/python

import collections

# Count the number of occurence of the most frequent part of speech in the list.
# Return    pos_ this frequent POS, 
#           pos_percent the proportion of this POS in the pos_list 
#           n_pos size of the source list.
def getMostFrequentPOS( pos_list ):
    "Count the percent of occurences of the most frequent part of speech in the list"

    n_pos = len(pos_list)
    if 0 == n_pos:
        return
   
    counter = collections.Counter(pos_list)
    # print(counter.most_common(1))

    #      n is number of pos_ in the pos_list
    (pos_, n) = next(iter( counter.most_common(1)))

    pos_percent = n / n_pos * 100

    return pos_, pos_percent, n_pos;
