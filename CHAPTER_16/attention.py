#!/usr/bin/env python3

"""
    Generate a Attention Plot
"""
import os
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow_hub as hub 
import tensorflow as tf
import math 

URL = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1'
module_embedd = hub.load(URL)

a = "are you still at home ?".split(' ')
y_labels = a + []

for i in range(len(a)):
    a[i] = module_embedd([a[i]]).numpy().tolist()[0]

a = np.array(a)
similarity_matrix = np.inner(a, a)

fig = plt.figure()
ax = plt.subplot(2,1,1)
ax.set_title('Attention: Similarity Matrix', fontsize=12)
ax.set_xticklabels(y_labels)
ax.set_yticklabels(y_labels)
mshow = ax.matshow(similarity_matrix)
fig.colorbar(mshow)

b = tf.matmul( tf.nn.softmax(similarity_matrix), a) 
b = b / math.sqrt(50)
ax = plt.subplot(2,1,2)
ax.set_title('Attention: Dot Product Attention',   fontsize=12  )
ax.set_xlabel('embedd_dim')
ax.set_ylabel('word')
mshow = ax.matshow(b)
fig.colorbar(mshow)
plt.show()