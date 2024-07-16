import glob
import os
import time
import json
from shutil import copyfile
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

def get_data_csv(type):
  df = None
  if type == 0:
    df = pd.read_csv("outputs.csv", dtype=str, na_filter=False)
  elif type == 1:
    df = pd.read_csv("outputs_backtranslate.csv", dtype=str, na_filter=False)
  elif type == 2:
    df = pd.read_csv("outputs_gpt.csv", dtype=str, na_filter=False)
  max_length = df.shape[0]

  if type == 1:
    end = df.loc[:, ['cve_id','desc','back_translate'] ]
  elif type == 2:
    end = df.loc[:, ['cve_id','desc','gpt_para'] ]
  
  X_data = df['desc'] if type == 0 else end
  print('Loaded data')
  return X_data[:max_length], df.loc[ : , df.columns != 'desc']
