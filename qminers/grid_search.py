import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from qminers.ndscaler import NDScaler

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from qminers.config import CONFIG


DEBUG = CONFIG['DEBUG']
SEQUENCE_LENGTH = CONFIG['SEQUENCE_LENGTH']
MODEL_PARAMS = CONFIG['MODEL_PARAMS']
EPOCHS = CONFIG['EPOCHS']

class BaselineRegressor(BaseEstimator, RegressorMixin):

    def __init__(self):
      pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        # return the last column
        # which in our case predicts the volume in next time-step will be equal to the volume in current time-step
        return X[:, -1]

def get_transformers():
  # return [MinMaxScaler(), RobustScaler(), StandardScaler()]
  return [MinMaxScaler()]


def get_classic_models():
  # first add baseline
  models = [BaselineRegressor()]

  lin_models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet()
  ]

  # add linear models
  models += lin_models

  if DEBUG:
    print('\n\t!!!get_classic_models in DEBUG mode !!!')

  else:
    svrs = [
      SVR(gamma='auto', kernel='linear'),
      SVR(gamma='auto', kernel='rbf')
    ]
    ensembles = [
      AdaBoostRegressor(),
      # AdaBoostRegressor(base_estimator=SVR(gamma='auto', kernel='linear')), # takes long
      BaggingRegressor(),
      BaggingRegressor(base_estimator=SVR(gamma='auto', kernel='linear')), # CV score=0.190 +/- 0.08
      BaggingRegressor(base_estimator=SVR(gamma='auto', kernel='rbf')), # CV score=0.168 +/- 0.06
      ExtraTreesRegressor(n_estimators=100),
      RandomForestRegressor(n_estimators=100),
      GradientBoostingRegressor(),
    ]

    # add SVRs and ensembles
    models += svrs + ensembles

  return models


def prepare_gridsearch_cv(n_splits, transformers_in, models, transformers_out):
  estimator = [('preprocessing', None), ('model', None)]
  estimator = Pipeline(estimator)

  regr = TransformedTargetRegressor(regressor=estimator,
                                     transformer=None)

  param_grid = {
      'regressor__preprocessing': transformers_in,
      'regressor__model': models,
      'transformer': transformers_out
  }

  grid_search = GridSearchCV(
      cv=n_splits,
      scoring=('r2', 'neg_mean_squared_error'),
      refit='r2',
      estimator=regr,
      param_grid=param_grid,
      return_train_score=False)

  return grid_search


def create_nn_model(model_params):
  inputs = Input(shape=(SEQUENCE_LENGTH, model_params['data_dim'],), name='input')

  output = model_params['layer_type'](model_params['output_dim'], return_sequences=False)(inputs)
  output = Dense(1)(output)

  model = Model(inputs=inputs, outputs=output)
  model.compile(loss=model_params['loss'],
                optimizer=Adam(lr=model_params['lr']),
                metrics=['mse'])
  return model


def fit_scaler(scaler_class, data):
  sc = NDScaler(scaler_class)
  sc.fit(data)
  return sc


def fit_scalers(scaler_class, data_x, data_y):
  return {'x': fit_scaler(scaler_class, data_x), 'y': fit_scaler(scaler_class, data_y)}


def transform_data(scalers, data_x, data_y):
  return scalers['x'].transform(data_x), scalers['y'].transform(data_y)


def transform_train_val_data(scalers, train_x, train_y, val_x, val_y):
  return transform_data(scalers, train_x, train_y), transform_data(scalers, val_x, val_y)


def select_fold(train, val, data):
  return (data[0][train], data[1][train]), (data[0][val], data[1][val])


def train_best_model(data_train, data_val, model_params, model_weights_path):
  # fit new scalers on train data
  scalers = fit_scalers(model_params['transformer_class'], *data_train)

  # transform data
  data_train, data_val = transform_train_val_data(scalers, *data_train, *data_val)

  # create model checkpointer to find the best model
  checkpointer = ModelCheckpoint(
      filepath=model_weights_path,
      verbose=0,
      save_best_only=True,
      monitor='val_mean_squared_error',
      save_weights_only=True
  )

  model = create_nn_model(model_params)

  model.fit(*data_train, epochs=EPOCHS, batch_size=model_params['batch_size'], verbose=0, validation_data=data_val, callbacks=[checkpointer])

  # load best model
  model.load_weights(model_weights_path)

  return scalers, model


def eval_model(model, model_params, scalers, data):
  # transform data
  data = transform_data(scalers, *data)

  res = model.evaluate(*data)

  # log results
  res = dict(zip([model_params['loss'], 'mse'], res))
  res['model_params'] = model_params
  res['scalers'] = scalers
  res['r2'] = evaluate_r2(model, *data)
  return res


def eval_fold(data_train, data_val, model_params, model_weights_path):
  # get the best model
  scalers, model = train_best_model(data_train, data_val, model_params, model_weights_path)

  # evaluate
  res = eval_model(model, model_params, scalers, data_val)

  return res


def eval_cv(n_splits, data, model_params, model_weights_path='/tmp/weights_%d.hdf5'):
  # Disable shuffle such that the train & val sequences don't overlap
  kf = KFold(n_splits=n_splits, shuffle=False)

  # prepare logging
  results = pd.DataFrame(index=np.arange(n_splits), columns=[model_params['loss'], 'r2', 'mse', 'model_params'])

  for i, (train, val) in enumerate(kf.split(data[1])):
    print("Running Fold", i+1, "/", n_splits)

    data_train, data_val = select_fold(train, val, data)

    results.loc[i, :] = eval_fold(data_train, data_val, model_params, model_weights_path % i)

  return results


def evaluate_r2(model, val_x, val_y):
  y_pred = model.predict(val_x)
  return r2_score(val_y, y_pred)


def get_default_nn_params(data_dim):
  model_params = MODEL_PARAMS.copy()
  model_params['data_dim'] = data_dim

  print('model loss: %s\n' % model_params['loss'])

  return model_params


def get_nn_models(data_dim):
  lstm_params = get_default_nn_params(data_dim)
  lstm_params['layer_type'] = LSTM
  lstm_params['transformer_class'] = MinMaxScaler

  # here we could create another nn model
  # and add it to the list
  models = [lstm_params]
  return models