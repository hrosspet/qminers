from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from qminers.config import CONFIG

TEST_RATIO = CONFIG['TEST_RATIO']
TIME_FORMAT = CONFIG['TIME_FORMAT']
DATA_PATH = CONFIG['DATA_PATH']
TRAIN_VAL_NAME = CONFIG['TRAIN_VAL_NAME']
TEST_NAME = CONFIG['TEST_NAME']
SEQUENCE_LENGTH = CONFIG['SEQUENCE_LENGTH']


def load_data():
  financial_df = pd.read_csv('data/sp500_2010_01_01-2014_07_31.csv')
  calendar_df = pd.read_csv('data/econ_calendar.csv')

  res = {
    'financial': financial_df,
    'calendar': calendar_df,
  }
  return res

def prepare_data_financial(data):
  financial_df = pd.DataFrame(data['financial'])
  # parse date
  financial_df['Date'] = financial_df['Date'].apply(pd.to_datetime)
  financial_df = financial_df.set_index('Date')
  return financial_df


def prepare_data_financial_seasonal(data):
  # process financial
  df = prepare_data_financial(data)

  # Day of month
  df['day'] = df.apply(lambda x: x.name.day, axis=1)
  df['dow'] = df.apply(lambda x: x.name.weekday(), axis=1)

  # AChampion's anwer: https://stackoverflow.com/questions/44124436/python-datetime-to-season/44124490
  df['season'] = (df.index.month % 12 + 3) // 3

  # move volume to the last position
  temp_vol = df.pop('Volume')
  df['Volume'] = temp_vol

  return df

'''
Take the raw financial data
and a subset of raw economic calendar data
- v1: only impact
and merge them together
'''
def prepare_data_financial_seasonal_economic(data):
  df = prepare_data_financial_seasonal(data)

  calendar_df = data['calendar']

  # process economic calendar
  if isinstance(data['calendar']['timestamp'].iloc[0], str):
    calendar_df['timestamp'] = calendar_df['timestamp'].apply(lambda x: datetime.strptime(x, TIME_FORMAT))

  calendar_df = calendar_df.set_index('timestamp')

  calendar_df = pd.get_dummies(calendar_df['impact'])

  # extract date so we can groupby
  calendar_df['date'] = calendar_df.apply(lambda x: x.name.date(), axis=1)

  calendar_df = calendar_df.groupby('date').sum()

  df = df.merge(calendar_df, left_index=True, right_index=True, how='left')

  # move volume to the last position
  temp_vol = df.pop('Volume')
  df['Volume'] = temp_vol

  return df

def split_data_in_out_classic(data):
  data = data.astype(float)
  return data.iloc[:-1, :], data.iloc[1:, -1]

def split_data_in_out_NN(data, batch_size=None, shuffle=False):
  if isinstance(data, pd.DataFrame):
    data = data.values

  if batch_size is None:
    # if batch_size undefined, transform the whole data
    batch_size = data.shape[0] - SEQUENCE_LENGTH

  batch_x = np.zeros((batch_size, SEQUENCE_LENGTH, data.shape[1]))
  batch_y = np.zeros((batch_size, 1))

  if shuffle:
    # random sample from training data
    rand_inds = np.random.randint(0, data.shape[0] - SEQUENCE_LENGTH - 1, batch_size)
  else:
    # not random, chronological

    # shift by 1 timestep
    step = 1

    stop_ind = min(batch_size * step - 1, data.shape[0] - SEQUENCE_LENGTH - 1)
    rand_inds = np.arange(start=0, step=step, stop=stop_ind)

  for i, idx in enumerate(rand_inds):
    end_idx = idx + SEQUENCE_LENGTH
    batch_x[i] = data[idx:end_idx,:]
    batch_y[i] = data[end_idx, -1]

  return batch_x, batch_y

def prepare_data(data, preparation_function, model_family='classic'):
  if model_family == 'classic':
    data_splitting_function = split_data_in_out_classic
    print('Data preparation for classic ML models')
  elif model_family == 'nn':
    data_splitting_function = split_data_in_out_NN
    print('Data preparation for NN models')
  else:
    raise RuntimeError('Unknown model family: %s' % model_family)

  data = preparation_function(data)
  data = train_test_split(data, test_size=TEST_RATIO, shuffle=False)

  res = {}
  splits = ['train_val', 'test']

  for s, d in zip(splits, data):
    res[s] = data_splitting_function(d)

  return res


def save_data(data_train_val, data_test, version):
  path = DATA_PATH + version
  if not os.path.isdir(path):
    os.mkdir(path)

  data_train_val.to_csv(path + TRAIN_VAL_NAME)
  data_test.to_csv(path + TEST_NAME)

  print('data saved to ' + path)


def process_results(results, data_preparation_functions):

  fig, ax = plt.subplots(len(data_preparation_functions), 2, figsize=(10, 10))

  for i, data_preparation_function in enumerate(data_preparation_functions):
    name = data_preparation_function.__name__

    r2 = np.concatenate((results[name]['classic']['r2'], results[name]['nn']['r2']), axis=1)
    mse = np.concatenate((results[name]['classic']['mse'], results[name]['nn']['mse']), axis=1)

    ax[i, 0].boxplot(r2)
    ax[i, 0].set_title(name + ' r2')
    ax[i, 0].set_ylim([-2.5, 1])

    ax[i, 1].boxplot(mse)
    ax[i, 1].set_title(name + ' mse');
    ax[i, 1].set_ylim([0, 5e16])

    plt.tight_layout()

  plt.savefig('results.png')
