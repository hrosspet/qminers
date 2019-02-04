from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from qminers.config import CONFIG

TEST_RATIO = CONFIG['TEST_RATIO']
TIME_FORMAT = CONFIG['TIME_FORMAT']
DATA_PATH = CONFIG['DATA_PATH']
TRAIN_VAL_NAME = CONFIG['TRAIN_VAL_NAME']
TEST_NAME = CONFIG['TEST_NAME']

'''
Take the raw financial data
and a subset of raw economic calendar data
- v1: only impact
and merge them together
'''
def prepare_data_v1(financial_df, calendar_df):
  # process financial
  financial_df['Date'] = financial_df['Date'].apply(pd.to_datetime)
  financial_df = financial_df.set_index('Date')

  # process economic calendar
  calendar_df['timestamp'] = calendar_df['timestamp'].apply(lambda x: datetime.strptime(x, TIME_FORMAT))
  calendar_df = calendar_df.set_index('timestamp')

  calendar_df = pd.get_dummies(calendar_df['impact'])

  # extract date so we can groupby
  calendar_df['date'] = calendar_df.apply(lambda x: x.name.date(), axis=1)

  calendar_df = calendar_df.groupby('date').sum()

  data = financial_df.merge(calendar_df, left_index=True, right_index=True, how='left')

  # move volume to the last position
  temp_vol = data.pop('Volume')
  data['Volume'] = temp_vol

  data_train_val, data_test = train_test_split(data, test_size=TEST_RATIO, shuffle=False)

  return data_train_val, data_test


def save_data(data_train_val, data_test, version):
  path = DATA_PATH + version
  if not os.path.isdir(path):
    os.mkdir(path)

  data_train_val.to_csv(path + TRAIN_VAL_NAME)
  data_test.to_csv(path + TEST_NAME)

  print('data saved to ' + path)


def load_data(version):
  path = DATA_PATH + version

  data_train_val = pd.read_csv(path + TRAIN_VAL_NAME)
  data_test = pd.read_csv(path + TEST_NAME)

  print('data loaded from ' + path)

  return data_train_val, data_test


def get_data_generator(data, batch_size, sequence_length, training=True):
  batch_x = np.zeros((batch_size, sequence_length, data_train_transformed.shape[1]))
  if return_sequences:
    batch_y = np.zeros((batch_size, sequence_length, 1))
  else:
    batch_y = np.zeros((batch_size, 1))

  # -1 because outputs are shifted by +1
  if training:
    # random sample from training data
    rand_inds = np.random.randint(0, data.shape[0] - sequence_length - 1, batch_size)
  else:
    # not random, chronological
    rand_inds = np.arange(start=0, step=sequence_length, stop=data.shape[0] - sequence_length - 1)

  while True:
    for i, idx in enumerate(rand_inds):
      end_idx = idx + sequence_length
      batch_x[i] = data[idx:end_idx,:]

      if return_sequences:
        batch_y[i] = data[idx + 1:end_idx + 1, -1, np.newaxis]
      else:
        batch_y[i] = data[end_idx, -1]

    yield batch_x, batch_y
