#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

from fastText import load_model
from fastText import util
import argparse
import numpy as np

if __name__ == "__main__":
   
	#initial creation of word vector
	#should do some pre-processing (e.g. lowercase, stopwords, punc etc before that)
	f=fastText.FastText.train_unsupervised('input.txt',
	 lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5,
	  wordNgrams=1, loss='softmax', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001,
	   label='__label__', verbose=2, pretrainedVectors='')
	
	#save the model for future use
	f.save_model('saved_model')
	
	#reload model later without having to train again
	f = load_model('saved_model') 
	
	#some stuff
	# Gets words with associated frequency sorted by default by descending order
	words, freq = f.get_words(include_freq=True) #freq not used
	print(words)
	print(freq)
	print(f.get_word_vector('rules'))	
	print(f.get_sentence_vector('rules.come and go'))
	
	#getting nearest neighbors
	
    # Retrieve list of normalized word vectors for the first words up
    # until the threshold count.
  
    # Gets words with associated frequeny sorted by default by descending order
    #words, freq = f.get_words(include_freq=True) #freq not used
    #words = words[:args.threshold]
    
    vectors = np.zeros((len(words), f.get_dimension()), dtype=float)
    for i in range(len(words)):
        wv = f.get_word_vector(words[i])
        wv = wv / np.linalg.norm(wv)
        vectors[i] = wv
    
    query = f.get_word_vector('secretary')
    
    nn1 = util.find_nearest_neighbor(
            query, vectors, ban_set=set(), cossims=None
        )
    print(nn1)
    print(f.get_words())
