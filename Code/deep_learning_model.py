from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.layers import Embedding, Conv1D, Dense, LSTM, MaxPooling1D, Flatten, Bidirectional
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
import numpy as np
import keras.backend as K

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import regularizers

import math
import time


optimizer = optimizers.Adam(learning_rate=0.001)

def validate_batch_sequential(x_val, y_val, model, fold, binary):

  start_time = time.time()

  y_pred = np.round(model.predict(x_val, batch_size=64))


  cur_y_pred = np.argmax(y_pred,axis=-1)
  cur_y_true = y_val.copy()

  perf_str = [f1_score(cur_y_true, cur_y_pred, average='macro')
    matthews_corrcoef(cur_y_true, cur_y_pred)]

  _val_metric = perf_str[-1]

  val_time = time.time() - start_time

  print("Validation time:", val_time, "s.\n")

  return _val_metric

class MyMetricsSequential(Callback):
  def __init__(self, x_val, y_val, fold, binary=True, patience_lr=3, patience_stopping=5,
          decrease_ratio=0.2, best_model_name='best_model.h5',
          **kwargs):
    super(Callback, self).__init__(**kwargs)
    print('x'*10)
    print(x_val.shape)
    print('x'*10)
    print(y_val.shape)
    print('x'*10)

    self.x_val = x_val

    self.y_val = y_val

    self.fold = fold

    self.binary = binary

    self.patience_lr = patience_lr
    self.patience_stopping = patience_stopping

    self.best_val = -1
    self.best_epoch_val = 0

    self.best_train = 100
    self.best_epoch_train = 0

    self.decrease_ratio = decrease_ratio
    self.best_model_name = best_model_name

  def on_train_begin(self, logs={}):
    self.val_f1s = []

  def on_epoch_end(self, epoch, logs={}):

    _val_metric = validate_batch_sequential(self.x_val, self.y_val, self.model, self.fold,
                            self.binary)

    cur_train_loss = float(logs.get('loss'))

    if cur_train_loss < self.best_train:
      self.best_train = cur_train_loss
      self.best_epoch_train = epoch

    self.val_f1s.append(_val_metric)

    if _val_metric > self.best_val:
      print("Val metric increases from", self.best_val, "to", _val_metric)
      print("Saving best model at epoch", epoch + 1)
      self.best_epoch_val = epoch
      self.best_val = _val_metric
      self.model.save(self.best_model_name)
    else:
      print("Val metric did not increase from the last epoch.")

    if epoch - self.best_epoch_train == self.patience_lr:
      self.model.optimizer.lr.assign(self.model.optimizer.lr * self.decrease_ratio)
      print("Train loss did not decrease in the last", self.patience_lr, "epochs, thus reducing learning rate to",
          K.eval(self.model.optimizer.lr), ".")

    if epoch - self.best_epoch_val == self.patience_stopping:
      print("Val metric did not increase in the last", self.patience_stopping,
          "epochs, thus stopping training.")
      self.model.stop_training = True

    return



class CNN:
  def __init__(self,x,num_of_class, callback, best_model_name):
    weight_decay = 0.00001
    self.best_model_name = best_model_name
    self.model = Sequential()
    
    self.model = Sequential()
    
    self.model.add(Embedding(20001, 300, input_length=x.shape[1]))
    
    # Trying different filters: 64, 128, 256
    # Trying different filter sizes: 1, 3, 5
    self.model.add(Conv1D(128, 3, padding='same', activation='relu', strides=1))
    
    self.model.add(MaxPooling1D(3))
    self.model.add(Flatten())
    self.model.add(Dense(128, activation='relu'))
    
    self.model.add(Dense(num_of_class, activation='softmax'))
    self.model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['acc'])
    self.le = LabelEncoder()
    self.call_back = callback

  def fit(self,x,y):
    print('*'*10)
    print(x.shape)
    print('*'*10)
    print(y.shape)
    print('*'*10)
    y = self.le.fit_transform(y)
    y = to_categorical(y)
    print('*'*10)
    print(y.shape)
    print('*'*10)
    # Train the model
    
    self.model.fit(x, y, batch_size=64, epochs=10, callbacks=[self.call_back])
    # Load best model
    self.model = load_model(self.best_model_name, compile=False)


  def predict(self,x):
    y_pred = self.model.predict(x, batch_size=64)
    # classes = np.argmax(y_pred,axis=1)
    return np.round_(np.argmax(y_pred, axis=-1))

  def evaluate(self,x,y):
    y = self.le.transform(y)
    y = to_categorical(y)
    accr = self.model.evaluate(x,y)
    print(f'Test set\n  Loss: {accr[0]}\n  MSE: {accr[1]}')
  
class LSTM_model:

  def __init__(self,x,num_of_class,callback, best_model_name):
    self.best_model_name = best_model_name
    self.model = Sequential()
    
    self.model.add(Embedding(20001, 300, input_length=x.shape[1]))
    
    # Trying different units: 64, 128, 256
    self.model.add(LSTM(units=128, dropout=0.2, recurrent_activation='sigmoid'))
    self.model.add(Dense(128, activation='relu'))
    
    self.model.add(Dense(num_of_class, activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    self.le = LabelEncoder()
    self.call_back = callback

  def fit(self,x,y):
    
    y = self.le.fit_transform(y)
    y = to_categorical(y)

    # Train the model
    
    self.model.fit(x, y, epochs=10, batch_size = 32, callbacks=[self.call_back])
    # Load best model
    self.model = load_model(self.best_model_name, compile=False)


  def predict(self,x):
    # padded_sequences = self.transform(x)
    y_pred = self.model.predict(x, batch_size = 32)
    return np.argmax(y_pred,axis=-1)
  
  def evaluate(self,x,y):
    y = self.le.transform(y)
    y = to_categorical(y)
    accr = self.model.evaluate(x,y)
    print(f'Test set\n  Loss: {accr[0]}\n  MSE: {accr[1]}')