import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import warnings  
warnings.filterwarnings('ignore')

# importing packages
import pandas as pd
import re
import numpy as np

# gensim
from gensim.models import FastText

# keras
import numpy as np
import tensorflow as tf

train_df = pd.read_csv("../dataset/train.csv")
dev_df = pd.read_csv("../dataset/dev.csv")
test_df = pd.read_csv("../dataset/test.csv")

train_seqs = [_ for _ in train_df["text_a"]]
train_labels = np.array([1 if (label == "yes") else 0 for label in train_df["label"]])

dev_seqs = [_ for _ in dev_df["text_a"]]
dev_labels = np.array([1 if (label == "yes") else 0 for label in dev_df["label"]])

test_seqs = [_ for _ in test_df["text_a"]]
test_labels = np.array([1 if (label == "yes") else 0 for label in test_df["label"]])

for i, seq in enumerate(train_seqs):
  train_seqs[i] = seq.split(" ")

for i, seq in enumerate(dev_seqs):
  dev_seqs[i] = seq.split(" ")

for i, seq in enumerate(test_seqs):
  test_seqs[i] = seq.split(" ")

# prepare data

vocab = []
  
for seq in train_seqs:
  for token in seq:
    if token not in vocab:
      vocab.append(token)

for seq in dev_seqs:
  for token in seq:
    if token not in vocab:
      vocab.append(token)

for seq in test_seqs:
  for token in seq:
    if token not in vocab:
      vocab.append(token)

def vocab_generator(vocab):
  yield vocab

# train model on train set
vec_model = FastText(size = 100, min_count = 1, window = 5)
vec_model.build_vocab(vocab_generator(vocab))
vec_model.train(vocab_generator(vocab), total_examples=vec_model.corpus_count, epochs=vec_model.iter)

train_data = [vec_model[seq] for seq in train_seqs]
dev_data = [vec_model[seq] for seq in dev_seqs]
test_data = [vec_model[seq] for seq in test_seqs]

def padData(seqs, target_size, vector_dim):
  new_list = np.zeros((len(seqs), target_size, vector_dim))
  
  for i, seq in enumerate(seqs):
    for j in range(min(target_size, len(seq))):
      new_list[i][j] = seq[j]

  return new_list

train_data_padded = padData(train_data, 124, 100)
dev_data_padded = padData(dev_data, 124, 100)
test_data_padded = padData(test_data, 124, 100)

train_data_padded = train_data_padded.reshape(len(train_labels), 124, 100)
dev_data_padded = dev_data_padded.reshape(len(dev_labels), 124, 100)
test_data_padded = test_data_padded.reshape(len(test_labels), 124, 100)

# rnn with vectors as inputs
dl_model = tf.keras.Sequential()
# dl_model.add(layers.Embedding(input_dim=100, output_dim=64, mask_zero=True))
dl_model.add(tf.keras.layers.LSTM(128, input_shape=(124, 100)))
dl_model.add(tf.keras.layers.Dense(64, activation='relu'))
dl_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

dl_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = dl_model.fit(x=train_data_padded, y=train_labels, epochs=10, validation_data=(dev_data_padded, dev_labels), validation_steps=10)

# validate results

results = dl_model.evaluate(test_data_padded, test_labels, batch_size=128)
print("Test loss:", results[0])
print("Test accuracy:", results[1])

