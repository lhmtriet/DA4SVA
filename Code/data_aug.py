import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw
import pandas as pd
import numpy as np

import os
import re
import random
import time

from parrot import Parrot

os.environ["MODEL_DIR"] = './model'


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)

def generate_set(x, y, num, seed = 10):
  random.seed(seed)
  output = []
  for j in range(num):
    i = random.randint(0, len(x)-1)
    output.append(x[i])
  return output, [y] * len(output)


def tfidf_aug(train_x, train_y, action = 'insert'):

  print(f"Augment size: {train_x.shape}")

  aug = None
  # Insert word by TF-IDF similarity
  if action == 'delete':
    # Delete word randomly
    aug = naw.RandomWordAug()
  else:
    aug = naw.TfIdfAug(
      model_path=os.environ.get("MODEL_DIR"),
      action = action,
      aug_p = 0.2
    )

  gen_x, gen_y = get_imbalance_data(train_x, train_y)

  print(f"Augment size: {gen_x.shape}")

  aug_df = gen_x.apply(lambda x: aug.augment(x, num_thread= 2)[0])

  print(aug_df)

  return np.concatenate((train_x, aug_df), axis=0), np.concatenate((train_y, gen_y), axis=0)
  # return train_x, train_y
  
def gen_mix(data, labels, num_of_run = 10000):
  set = {}
  for i, e in enumerate(data):
    if labels[i] not in set:
        set[labels[i]] = []
    set[labels[i]].append(e)
  
  m = 0
  for k in set.keys():
    if m < len(set[k]):
      m = len(set[k])
    print(f'Key: {k} - {len(set[k])}')

  out_x = pd.Series(dtype='string')
  out_y = pd.Series(dtype='string')
  for k in set.keys():
    x, y = generate_set(set[k],k,num_of_run)

    out_x = out_x.append(x)
    out_y = out_y.append(y)
  return out_x, out_y

def get_imbalance_data(data, labels):
  set = {}
  for i, e in enumerate(data):
    if labels[i] not in set:
        set[labels[i]] = []
    set[labels[i]].append(e)
  
  m = 0
  for k in set.keys():
    if m < len(set[k]):
      m = len(set[k])

  out_x = []
  out_y = []
  for k in set.keys():
    if m > len(set[k]):
      x, y = generate_set(set[k],k,m - len(set[k]))
      out_x += x
      out_y += y

  return pd.Series(out_x), pd.Series(out_y)

def tfidf_aug_mix(train_x, train_y):
  # Tokenize input
  train_x_tokens = [_tokenizer(x) for x in train_x]

  output = []

  # Train TF-IDF model
  tfidf_model = nmw.TfIdf()
  tfidf_model.train(train_x_tokens)
  tfidf_model.save(os.environ.get("MODEL_DIR"))

  print(f"Augment size: {train_x.shape}")


  gen_x, gen_y = get_imbalance_data(train_x, train_y)

  # Insert word by TF-IDF similarity
  aug = naw.TfIdfAug(
      model_path=os.environ.get("MODEL_DIR"),
      action = 'insert',
      aug_p = 0.2
    )

  aug_df = gen_x.apply(lambda x: aug.augment(x, num_thread= 2)[0])

  output = np.concatenate((train_x, aug_df), axis=0), np.concatenate((train_y, gen_y), axis=0)

  # Subtitute word by TF-IDF similarity
  aug = naw.TfIdfAug(
      model_path=os.environ.get("MODEL_DIR"),
      action = 'substitute',
      aug_p = 0.2
    )

  aug_df = gen_x.apply(lambda x: aug.augment(x, num_thread= 2)[0])

  output = np.concatenate((train_x, aug_df), axis=0), np.concatenate((train_y, gen_y), axis=0)

  # Delete word by TF-IDF similarity
  aug = naw.RandomWordAug()

  aug_df = gen_x.apply(lambda x: aug.augment(x, num_thread= 2)[0])


  output = np.concatenate((train_x, aug_df), axis=0), np.concatenate((train_y, gen_y), axis=0)

  return output

def get_aug(action):
  if action == 'delete':
    # Delete word randomly
    return naw.RandomWordAug()
  elif action == 'synonym':
    return naw.SynonymAug()

  else:
    return naw.TfIdfAug(
      model_path=os.environ.get("MODEL_DIR"),
      action = action,
      aug_p = 0.2
    )


class data_augmentation:
  def __init__(self, action):
    if action == 'ISD':
      self.actions = [get_aug('insert'), get_aug('substitute'), get_aug('delete')]
    else:
      self.actions = [get_aug(action)]  
  def fit_resample(self, train_x, train_y):
      # Tokenize input
      train_x_tokens = [_tokenizer(x) for x in train_x]

      output = []

      # Train TF-IDF model
      tfidf_model = nmw.TfIdf()
      tfidf_model.train(train_x_tokens)
      tfidf_model.save(os.environ.get("MODEL_DIR"))

      print(f"Augment size: {train_x.shape}")

      gen_x, gen_y = get_imbalance_data(train_x, train_y)

      for aug in self.actions:
        gen_x = gen_x.apply(lambda x: aug.augment(x, num_thread= 2)[0])
      output = np.concatenate((train_x, gen_x), axis=0), np.concatenate((train_y, gen_y), axis=0)
      return output


class back_translation:
  def __init__(self):
    self.aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-de',
    to_model_name='Helsinki-NLP/opus-mt-en-de')

  def check_aug(self, x):
    print(f'Start Augmentation: {x}')
    results = self.aug.augment(x)
    print(f'result {results}')
    start = time.time()
    res = results[0]
    end = time.time()
      
    print(f'Aug: {res}. \n Time run  is: {end-start} s.')
    return res
  
  def gen_sample(self, train_x, train_y):
      output = []
      gen_x, gen_y = get_imbalance_data(train_x, train_y)
      aug_df = gen_x.apply(lambda x: self.check_aug(x))
      output = np.concatenate((train_x, aug_df), axis=0), np.concatenate((train_y, gen_y), axis=0)
      return output
  
  def fit_resample(self, train_x, train_y):
      output = []
      origin_desc_x = train_x['desc']
      gen_x, gen_y = get_imbalance_data(train_x['back_translate'], train_y)
      output = np.concatenate((origin_desc_x, gen_x), axis=0), np.concatenate((train_y, gen_y), axis=0)
      return output


class paraphrase_gpt:
  def __init__(self):
    self.aug = None

  def generate_set(self, x, y, num):
    output = []
    for j in range(num):
      i = random.randint(0, len(x)-1)
      split_result = x[i].split('thisisasplitter')
      k = random.randint(0, len(split_result)-1)
      output.append(split_result[k])
    return output, [y] * len(output)

  def get_imbalance_data(self, data, labels):    
    # Get minority class
    set_data = {}
    for i, e in enumerate(data):
      if labels[i] not in set_data:
        set_data[labels[i]] = []
      set_data[labels[i]].append(e)

    # Check majority class
    m = 0
    for k in set_data.keys():
      if m < len(set_data[k]):
        m = len(set_data[k])

    # Check majority class
    out_x = []
    out_y = []
    for k in set_data.keys():
      if m > len(set_data[k]):
        x, y = self.generate_set(set_data[k],k,m - len(set_data[k]))
        out_x += x
        out_y += y
    return np.array(out_x), np.array(out_y)
    
  def fit_resample(self, train_x, train_y):
      output = []
      origin_desc_x = train_x['desc']

      gen_x, gen_y = self.get_imbalance_data(train_x['gpt_para'], train_y)

      output = np.concatenate((origin_desc_x, gen_x), axis=0), np.concatenate((train_y, gen_y), axis=0)
      return output
