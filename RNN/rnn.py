# -*- coding: utf-8 -*-
"""RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NGvwRQmTcqWWIIHEMX7-bpgjvbr7ynS0
"""

!pip install --user gensim

!pip install -U numpy==1.18.5

import numpy as np
import pandas as pd

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

# train model on vocab
vec_model = FastText(size = 100, min_count = 1, window = 5)
vec_model.build_vocab(vocab_generator(vocab))
vec_model.train(vocab_generator(vocab), total_examples=vec_model.corpus_count, epochs=vec_model.iter)

train_data = [vec_model[seq] for seq in train_seqs]
dev_data = [vec_model[seq] for seq in dev_seqs]
test_data = [vec_model[seq] for seq in test_seqs]

def padAndScaleData(seqs, target_size, vector_dim):
  new_list = np.zeros((len(seqs), target_size, vector_dim))
  
  for i, seq in enumerate(seqs):
    for j in range(min(target_size, len(seq))):
      new_list[i][j] = 1000*seq[j]

  return new_list

max_seq_length = 124

train_data_padded = padAndScaleData(train_data, max_seq_length, 100)
dev_data_padded = padAndScaleData(dev_data, max_seq_length, 100)
test_data_padded = padAndScaleData(test_data, max_seq_length, 100)

train_data_padded = train_data_padded.reshape(len(train_labels), max_seq_length, 100)
dev_data_padded = dev_data_padded.reshape(len(dev_labels), max_seq_length, 100)
test_data_padded = test_data_padded.reshape(len(test_labels), max_seq_length, 100)

# rnn with vectors as inputs
dl_model = tf.keras.Sequential()
# dl_model.add(layers.Embedding(input_dim=100, output_dim=64, mask_zero=True))
dl_model.add(tf.keras.layers.LSTM(128, input_shape=(max_seq_length, 100)))
dl_model.add(tf.keras.layers.Dense(64, activation='relu'))
dl_model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
dl_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

dl_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = dl_model.fit(x=train_data_padded, y=train_labels, epochs=10, validation_data=(dev_data_padded, dev_labels), validation_steps=10)

# validate results

results = dl_model.evaluate(test_data_padded, test_labels, batch_size=128)
print("Test loss:", results[0])
print("Test accuracy:", results[1])
