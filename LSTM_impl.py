import os
import gc
import time
import math
import datetime
from math import log, floor
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from tqdm.notebook import tqdm as tqdm

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import sklearn

import warnings
warnings.filterwarnings("ignore")

#Model adapted from here: https://www.kaggle.com/code/gopidurgaprasad/m5-forecasting-eda-lstm-pytorch-modeling/notebook


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def read_data(PATH):
    print('Reading files...')
    calendar = pd.read_csv(f'{PATH}/calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(f'{PATH}/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(f'{PATH}/sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(f'{PATH}/sample_submission.csv')
    sales_train_evaluation = pd.read_csv(f'{PATH}/sales_train_evaluation.csv')
    print('Sales train evaluation has {} rows and {} columns'.format(sales_train_evaluation.shape[0], sales_train_evaluation.shape[1]))

    return calendar, sell_prices, sales_train_validation, submission, sales_train_evaluation

calendar, sell_prices, sales_train_validation, submission, sales_train_evaluation = read_data("./")


sales_train_validation_melt = pd.melt(sales_train_evaluation, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='day', value_name='demand')

loss_dict = {}

!mkdir "item_files"

class LSTM(nn.Module):
    def __init__(self, input_size=3062, hidden_layer_size=100, output_size=28):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self, input_seq):

        lstm_out, self.hidden_cell = self.lstm(input_seq)

        lstm_out = lstm_out[:, -1]

        predictions = self.linear(lstm_out)

        return predictions

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 128
EPOCHS = 4
start_e = 1


model = LSTM()
model.to(DEVICE)
total_train_loss = []
for store in ["TX_1"]: #, "CA_2", "CA_3", "CA_4", "WI_1", "WI_2", "WI_3", "TX_1", "TX_2", "TX_3" ]:

  train_loss = []

  print('Merge')
  sales_store_id = sales_train_validation_melt[sales_train_validation_melt.store_id == store]
  new_store_id = pd.merge(sales_store_id, calendar, left_on="day", right_on="d", how="left")
  new_store_id = pd.merge(new_store_id, sell_prices, left_on=["store_id", "item_id", "wm_yr_wk"],right_on=["store_id", "item_id", "wm_yr_wk"], how="left")
  new_store_id["day_int"] = new_store_id.day.apply(lambda x: int(x.split("_")[-1]))

  del sales_store_id
  gc.collect()

  store_id = new_store_id

  del new_store_id
  gc.collect()

  print('fillna')
  store_id = store_id[["item_id","day_int", "demand", "sell_price", "date"]]
  store_id.fillna(0, inplace=True)
  print(store_id.shape)


  def date_features(df):
      
      df["date"] = pd.to_datetime(df["date"])
      df["day"] = df.date.dt.day
      df["month"] = df.date.dt.month
      df["week_day"] = df.date.dt.weekday

      df.drop(columns="date", inplace=True)

      return df

  def sales_features(df):

      df.sell_price.fillna(0, inplace=True)

      return df

  def rolling_data_features(df):

      df["lag_t28"] = df["demand"].transform(lambda x: x.shift(28))
      df['rolling_mean_t180'] = df['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
      df['rolling_mean_t90'] = df['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
      df['rolling_mean_t60'] = df['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
      df['rolling_mean_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
      df["rolling_mean_t7"] = df["demand"].transform(lambda x:x.shift(28).rolling(7).mean())
      df['rolling_std_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).std())
      df['rolling_std_t7'] = df['demand'].transform(lambda x: x.shift(28).rolling(7).std())

      df.fillna(0, inplace=True)

      return df


  for item in tqdm(store_id.item_id.unique()):
      one_item = store_id[store_id.item_id == item][["demand", "sell_price", "date"]]
      item_df = date_features(one_item)
      item_df = sales_features(item_df)
      item_df = rolling_data_features(item_df)
      joblib.dump(item_df.values, f"item_files/{item}.npy")

  data_info = store_id[["item_id", "day_int"]]

  del store_id
  gc.collect()

  train_df = data_info[(100 < data_info.day_int) & ( data_info.day_int < 1700)]

  valid_df = data_info[(1857 < data_info.day_int) & ( data_info.day_int < 1885)]

  test_df = data_info[data_info.day_int == 1913]


  label = preprocessing.LabelEncoder()
  label.fit(train_df.item_id)

  del data_info
  gc.collect()

  class DataLoading:
      def __init__(self, df, train_window = 28, predicting_window=28):
          self.df = df.values
          self.train_window = train_window
          self.predicting_window = predicting_window

      def __len__(self):
          return len(self.df)
      
      def __getitem__(self, item):
          df_item = self.df[item]
          item_id = df_item[0]
          day_int = df_item[1]
          
          item_npy = joblib.load(f"item_files/{item_id}.npy")
          item_npy_demand = item_npy[:,0]
          features = item_npy[day_int-self.train_window:day_int]
      

          predicted_demand = item_npy_demand[day_int:day_int+self.predicting_window]

          item_label = label.transform([item_id])
          onehot = [0] * 3049
          onehot[item_label[0]] = 1

          list_features = []
          for f in features:
              one_f = []
              one_f.extend(onehot)
              one_f.extend(f)
              list_features.append(one_f)

          return {
              "features" : torch.Tensor(list_features),
              "label" : torch.Tensor(predicted_demand)
          }

  def lossFunc(pred1, targets):
      l1 = nn.MSELoss()(pred1, targets)
      return l1

  def train_model(model,train_loader, epoch, optimizer):
      model.train()
      total_loss = 0
      
      t = tqdm(train_loader)
      
      for i, d in enumerate(t):
          
          item = d["features"].cuda().float()
          y_batch = d["label"].cuda().float()

          optimizer.zero_grad()

          out = model(item)
          loss = lossFunc(out, y_batch)

          total_loss += loss
          
          t.set_description(f'Epoch {epoch} : , LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

          loss_val = (total_loss/(i+1)).item()
          train_loss.append(loss_val)

          loss.backward()
          optimizer.step()

      return model
          

  def evaluate_model(model, val_loader, epoch, scheduler):
      model.eval()
      loss = 0
      RMSE_list = []
      with torch.no_grad():
          for i,d in enumerate(tqdm(val_loader)):
              item = d["features"].cuda().float()
              y_batch = d["label"].cuda().float()

              o1 = model(item)
              l1 = lossFunc(o1, y_batch)
              loss += l1
              
              o1 = o1.cpu().numpy()
              y_batch = y_batch.cpu().numpy()
              
              for pred, real in zip(o1, y_batch):
                  rmse = np.sqrt(sklearn.metrics.mean_squared_error(real, pred))
                  RMSE_list.append(rmse)

      loss /= len(val_loader)
      
      scheduler.step(loss)

      print(f'\n Eval loss: %.4f RMSE : %.4f'%(loss, np.mean(RMSE_list)))

      return loss
      

  

  traindata = DataLoading(train_df)

  train_loader = torch.utils.data.DataLoader(
      dataset=traindata,
      batch_size= TRAIN_BATCH_SIZE,
      shuffle=True,
      num_workers=4,
      drop_last=True
  )


  validdata = DataLoading(valid_df)

  valid_loader = torch.utils.data.DataLoader(
      dataset=validdata,
      batch_size= TRAIN_BATCH_SIZE,
      shuffle=False,
      num_workers=4,
      drop_last=True
  )

  testdata = DataLoading(test_df)
  test_loader = torch.utils.data.DataLoader(
      dataset=testdata,
      batch_size= TEST_BATCH_SIZE,
      shuffle=False,
      num_workers=4,
      drop_last=True
  )



  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, mode='min', factor=0.7, verbose=True, min_lr=1e-5)

  print('Begin training for store:', store)
  for epoch in range(start_e, EPOCHS+1):
      model = train_model(model, train_loader, epoch, optimizer)
      evaluate_model(model, valid_loader, epoch, scheduler=scheduler)

  print('Testing for store:', store)
  test_loss = evaluate_model(model, test_loader, epoch, scheduler=scheduler)


  del traindata, validdata, testdata
  gc.collect()

  fig, axs = plt.subplots(1)
  fig.suptitle(store + " train loss")

  x_vals = np.arange(len(train_loss))
  axs.plot(x_vals, train_loss)
