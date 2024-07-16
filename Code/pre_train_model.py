from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import torch
import time

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

class Tfidf:
  def __init__(self,data):
    self.model = TfidfVectorizer(min_df=0.001, analyzer='word')
    self.model.fit(data)
  def infer_vector(self,data):
    counts = self.model.transform(data)
    return pd.DataFrame(counts.A, columns=self.model.get_feature_names_out())

class doc2vec_convert:
  def __init__(self, data):
    self.tagged_data = [TaggedDocument(words=tokenize_text(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    self.model = Doc2Vec(vector_size=300, min_count=20, epochs=80, workers = 16) 
    self.model.build_vocab(self.tagged_data)
    self.model.train(self.tagged_data, total_examples=self.model.corpus_count, epochs=80)
  def infer_vector(self, data):
    out = []
    for x in data:
      res = self.model.infer_vector(tokenize_text(x.lower()))
      out.append(res)
    return np.array(out)


class DLFeature:
    def __init__(self,data):
      self.max_len = 5000
      self.max_words = 5000
      self.tokenizer = Tokenizer(num_words=self.max_words)
      self.tokenizer.fit_on_texts(data)

    def infer_vector(self,data):
      sequences = self.tokenizer.texts_to_sequences(data)
      out = pad_sequences(sequences, maxlen=self.max_len)
      return out