import sys
import os
from deep_learning_model import MyMetricsSequential
import pandas as pd
import numpy as np
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import ParameterSampler

from pre_train_model import Tfidf, doc2vec_convert, DLFeature
from deep_learning_model import LSTM_model, CNN

from sklearn.ensemble import RandomForestClassifier
from data_aug import data_augmentation, back_translation, paraphrase_gpt

import time

from utils import get_data_csv

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder


# Constaint value
table_name = 'cve'
n_most_common_words = 20000
max_len = 256
MAX_FEATURES = 20001

def dl_model_train(model, train_set, val_set, test_set, balancer, id, cve_id):
  train_X, train_y  = train_set
  val_X, val_y = val_set
  test_X, test_y = test_set

  pred_out = pd.DataFrame()
  pred_out['cve_id'] = cve_id
  pred_out['task'] = [id]*len(test_X)
  pred_out['desc'] = test_X
  
  if len(val_X.shape) > 1:
    val_X = val_X['desc']
    test_X = test_X['desc'] 

  if balancer != None and balancer[1] == 2:
    start = time.time()
    print(f'Before shape: {train_X.shape}')   
    train_X, train_y = balancer[0].fit_resample(np.array(train_X).reshape(-1,1), train_y) 
    end = time.time()
    train_X = pd.Series([x[0] for x in train_X ]) 
    print(f'Finished balancing: {end-start} s. After shape {balancer[1]}: {train_X.shape} - {train_y.shape}.')

  if balancer != None and balancer[1] == 1:
    start = time.time()
    print(f'Before shape: {train_X.shape}')  
    train_X, train_y = balancer[0].fit_resample(train_X, train_y)
    train_X = pd.Series(train_X)
    end = time.time()
    print(f'Finished balancing: {end-start} s. After shape {balancer[1]}: {train_X.shape} - {train_y.shape}.')

  tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
  tokenizer.fit_on_texts(train_X)

  # print(f'Max length: {tokenizer.word_counts}')
  print(f'Max length: {tokenizer.num_words}')

  sequences_train = tokenizer.texts_to_sequences(train_X)
  sequences_val = tokenizer.texts_to_sequences(val_X)
  sequences_test = tokenizer.texts_to_sequences(test_X)

  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))

  train_X = pad_sequences(sequences_train, maxlen=max_len)
  val_X = pad_sequences(sequences_val, maxlen=max_len)
  test_X = pad_sequences(sequences_test, maxlen=max_len)

  best_model_name = f'best_model_{id}.h5'

  callback =  MyMetricsSequential(val_X, val_y, fold=5, binary=False, patience_stopping=5, best_model_name= best_model_name)
  num_train_class = len(np.unique(train_y, return_counts= False))
  model_build = model(train_X,num_train_class, callback, best_model_name)

  model_build.fit(train_X, train_y)
  pred_y = model_build.predict(val_X)
  pred_test = model_build.predict(test_X)

  os.remove(best_model_name)

  
  pred_out['truth'] = test_y
  pred_out['pred'] = pred_test

  pred_out.to_csv(f'./pred_truth/{id}.csv')

  return [
      recall_score(val_y,pred_y, average='macro'), precision_score(val_y,pred_y, average='macro'), f1_score(val_y,pred_y, average='macro'), matthews_corrcoef(val_y,pred_y), 
      recall_score(test_y,pred_test, average='macro'), precision_score(test_y,pred_test, average='macro'), f1_score(test_y,pred_test, average='macro'), matthews_corrcoef(test_y,pred_test)
  ]

def dl_run(data, no_of_split ,model, output, balancer, out_set):
  X, y = data
  ids = y['cve_id']
  y = pd.factorize(y[output])[0]

  N = X.shape[0]
  chunks = N // no_of_split
  validation = N - chunks * 3
  test = N - chunks * 2
  i = 0
  records = []
  validation = chunks 
  test = chunks * 2
  while test+chunks <= N:
    print(f'Preparing chunk {i}')
    train_set = (X.iloc[:validation], y[:validation])
    val_set = (X.iloc[validation:test], y[validation:test])
    test_set = (X.iloc[test:test+chunks], y[test:test+chunks])
    print(f'Done preparing chunk {i} ...')
    out = [i] + out_set + ['-'] + dl_model_train(model,train_set, val_set, test_set, balancer,'_'.join(out_set),ids[test:test+chunks])
    validation += chunks
    test += chunks
    i += 1
    records.append(out)
  return records

def model_train_and_tune(model, params, train, val, test, pre_train, balancer = None):
  start_model = time.time()
  train_X, train_y = train
  val_X, val_y = val
  test_X, test_y = test

  le = LabelEncoder()
  le.fit(train_y) 
  num_train_class = len(np.unique(train_y, return_counts= False))
  num_val_class = len(np.unique(val_y, return_counts= False))
  num_test_class = len(np.unique(test_y, return_counts= False))
  train_y = le.transform(train_y)
  val_y = le.transform(val_y)
  test_y = le.transform(test_y)

  if len(val_X.shape) > 1:
    val_X = val_X['desc']
    test_X = test_X['desc']  

  if balancer != None and balancer[1] == 1:
    start = time.time()
    print(f'Before shape: {train_X.shape}')  
    train_X, train_y = balancer[0].fit_resample(train_X, train_y)
    train_X = pd.Series(train_X)
    end = time.time()
    print(f'Finished balancing: {end-start} s. After shape {balancer[1]}: {train_X.shape} - {train_y.shape}.')
    
  if balancer != None and balancer[1] == 2:
    start = time.time()
    print(f'Before shape: {train_X.shape}')   
    train_X, train_y = balancer[0].fit_resample(np.array(train_X).reshape(-1,1), train_y) 
    end = time.time()
    train_X = pd.Series([x[0] for x in train_X ]) 
    print(f'Finished balancing: {end-start} s. After shape {balancer[1]}: {train_X.shape} - {train_y.shape}.') 
  
  start = time.time()
  pre_process = pre_train(train_X)
  end = time.time()

  print(f'Finished feature applied: {end-start} s.')


  train_X = pre_process.infer_vector(train_X)
  val_X = pre_process.infer_vector(val_X)
  test_X = pre_process.infer_vector(test_X)

  # tunned_model = None

  best_score = -1
  best_grid = None

  output = []

  #tuning
  if params != None:
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    for g in ParameterSampler(params, n_iter=20):
      r,c = train_X.shape
      if ('n_neighbors' in g) and (int(g['n_neighbors']) >= train_X.shape[0]):
        continue
      val_model = model(num_train_class,r,c)
      val_model.set_params(**g)
      val_model.fit(train_X, train_y)
      pred_y = val_model.predict(val_X)
      pred_test = val_model.predict(test_X)
      curr_score = matthews_corrcoef(val_y,pred_y)
      output.append(
        (repr(g), 
         f1_score(val_y,pred_y, average='macro'), matthews_corrcoef(val_y,pred_y), 
         f1_score(test_y,pred_test, average='macro'), matthews_corrcoef(test_y,pred_test))
        )

      # save if best
      if curr_score > best_score:
        best_score = curr_score
        best_grid = g
        # tunned_model = model

    best_score = 0
    if best_grid != None:
      print(f'Best Hyperparameters: {best_grid}')
      test_model = model(num_train_class,r,c)
      test_model.set_params(**best_grid)
      test_model.fit(train_X, train_y)
      pred_test = test_model.predict(test_X)
      best_score = matthews_corrcoef(test_y,pred_test)

    end_model = time.time()
    print(f'Finished tuning models: {end_model-start_model} s.')
  else:
    model.fit(train_X, train_y)
    model.evaluate(val_X,val_y)
    model.evaluate(test_X,test_y)
    pred_y = model.predict(val_X)
    pred_test = model.predict(test_X)
    best_score = matthews_corrcoef(val_y,pred_y)
    output.append(
        ('-', 
         f1_score(val_y,pred_y, average='macro'), matthews_corrcoef(val_y,pred_y), 
         f1_score(test_y,pred_test, average='macro'), matthews_corrcoef(test_y,pred_test))
        )
  return output, best_score

def model_for_output(data, model, params, output, no_of_split, pre_train, balancer, out_set):
  X, y = data
  y = pd.factorize(y[output])[0]
  N = X.shape[0]
  chunks = N // no_of_split
  validation = chunks 
  test = chunks * 2

  i = 0
  sum_score = 0
  results = []
  while test+chunks <= N:
    print(f'Preparing chunk {i}')
    train_set = (X.iloc[:validation], y[:validation])
    val_set = (X.iloc[validation:test], y[validation:test])
    if ((test+chunks) <= N) :
      test_set = (X.iloc[test:N], y[test:N])
    else:
      test_set = (X.iloc[test:test+chunks], y[test:test+chunks])
      
    print(f'Done preparing chunk {i} ...')
    start = time.time()
    chunk_score, mcc = model_train_and_tune(model, params, train_set, val_set, test_set, pre_train, balancer)
    end = time.time()
    print(f'Chunk {i} MCC score: {mcc}. Time run for this chunk is: {end-start} s.')
    sum_score += mcc
    # validation -= chunks
    # test -= chunks
    # i -= 1
    validation += chunks
    test += chunks
    i += 1
    for score in chunk_score:
      result = []
      result[-1:-1] = out_set
      result.extend(score)
      result.insert(0,str(i))
      results.append(result)
        
  avg_score = sum_score / i if i > 0 else 0
  print(f'Chunk {i-1} average: {avg_score}')
  return results

def switch(dict, num):
  result = dict.get(num, None)
  return result

def save_result(name, result):
  fields = ['#','Feature', 'Model', 'DA', 'Output', 'HP', 'F1_val' ,'MCC_val', 'F1_test' ,'MCC_test']

  data = pd.DataFrame(result,columns=fields)

  data.to_csv(f'{name}.csv')

  return None

def main(f,da,m,out):
  # print(f'Start building {f} {da} {m} {out}')

  print(f,da,m,out)

  type = 1 if (da == 'DABackTrans') else 0
  type = 2 if (da == 'DAParaphraseGPT') else type
  X, y = get_data_csv(type)

  dict_feature = {
    'TFIDF': Tfidf,
    'Doc2Vec': doc2vec_convert,
    'DL': DLFeature
  }

  dict_DA ={
    'None': None,
    'OS'  : (RandomOverSampler(sampling_strategy='all'), 2),
    'US'  : (RandomUnderSampler(random_state=0),2),
    'DAI' : (data_augmentation('insert'),1),
    'DAS' : (data_augmentation('substitute'),1),
    'DAD' : (data_augmentation('delete'),1),
    'DASyn': (data_augmentation('synonym'),1),
    'Comb' : (data_augmentation('ISD'),1),
    'DABackTrans': (back_translation(),1),
    'DAParaphraseGPT': (paraphrase_gpt(),1),
  }

  max_iter = 10000
  workers = 16

  models = [
    {
      "name": "RandomForest",
      "model": lambda x,r,c: RandomForestClassifier(oob_score=True,max_depth=None, random_state=42, n_jobs=workers),
      "params": {
        'n_estimators': [100,300,500],
        'max_leaf_nodes': [100,200,300]
      }
    },
    {
      'name': "CNN",
      'model': CNN,
      'params': None
    },
    {
      'name': "LSTM",
      'model': LSTM_model,
      'params': None
    }
  ]

  dict_model = {
    'RF' : (models[3]['model'], models[3]['params']),
    'CNN' : (models[8]['model'], models[8]['params']),
    'LSTM' : (models[9]['model'], models[9]['params']),
  }
  

  balancer = switch(dict_DA,da)
  model, params = switch(dict_model,m)

  no_of_split = 5 # 3-round time-based evaluation
  pre_train = switch(dict_feature,f)

  if (f == 'DL'):
    # Check if GPUs are available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth for GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set TensorFlow to use only the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    print('GPU check done.')

    results = dl_run((X,y), no_of_split, model, out, balancer,[f,m,da,out])
  else:
    results = model_for_output((X,y), model, params, out, no_of_split, pre_train, balancer, [f,m,da,out])
  save_result(f'results_cvss2/{f}_{m}_{da}_{out}', results)
  print(f'Finished building {f}_{m}_{da}_{out}')


f = sys.argv[1]
da = sys.argv[2]
m = sys.argv[3]
out = sys.argv[4]
start = time.time()
main(f,da,m,out)
end = time.time()
print("The time of execution of the program is :", (end-start), "s")
