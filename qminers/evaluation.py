import numpy as np
from sklearn.model_selection import train_test_split

from qminers.grid_search import train_best_model, get_transformers, get_classic_models, prepare_gridsearch_cv, get_nn_models, eval_cv
from qminers.config import CONFIG

N_SPLITS = CONFIG['N_SPLITS']

def extract_res_cross_validation(cv_results_list, data):
  results = {
      'r2': [],
      'mse': [],
      'params': []
  }

  # python doesn't have min value for int -> select reasonably small
  max_r2 = -1000000
  for cv_result in cv_results_list:
    results['r2'].append(cv_result['r2'])
    results['mse'].append(cv_result['mse'])
    results['params'].append(cv_result['mse'])

    r2_mean = cv_result['r2'].mean()

    # find best estimator
    if r2_mean > max_r2:
      max_r2 = r2_mean
      # store the best params, so we can retrain it on the whole train_val
      results['best_params'] = cv_result['model_params'].iloc[0]

  results['r2'] = np.array(results['r2']).T
  results['mse'] = np.array(results['mse']).T

  # now retrain the best model

  # shuffle, because we want the best model given the data
  data_train_x, data_val_x, data_train_y, data_val_y = train_test_split(*data, test_size=1/N_SPLITS, shuffle=True)

  results['best_estimator'] = train_best_model((data_train_x, data_train_y), (data_val_x, data_val_y), results['best_params'], model_weights_path='/tmp/weights_top.hdf5')

  # in case something goes wrong, store the whole thing
  results['grid_search'] = cv_results_list

  return results

def extract_res_grid_search(grid_search):
  results = {}
  results['best_estimator'] = grid_search.best_estimator_

  cv_results_ = grid_search.cv_results_

  results['r2'] = np.array([cv_results_['split%d_test_r2' % i] for i in range(N_SPLITS)])
  results['mse'] = -np.array([cv_results_['split%d_test_neg_mean_squared_error' % i] for i in range(N_SPLITS)])
  results['params'] = grid_search.cv_results_['params']

  # in case something goes wrong, store the whole thing
  results['grid_search'] = grid_search

  return results

def eval_classic_ml_models(data, n_splits):
  # prepare transformers & models
  transformers_in = get_transformers()
  models = get_classic_models()
  transformers_out = get_transformers()

  # prepare gridsearch with cross-validation
  grid_search = prepare_gridsearch_cv(n_splits, transformers_in, models, transformers_out)

  # fit models via gridsearch
  grid_search.fit(*data)

  # process results
#   return pd.DataFrame(grid_search.cv_results_)
  results = extract_res_grid_search(grid_search)
  return results

def eval_nn_models(data, n_splits):
  data_dim = data[0].shape[2]

  models = get_nn_models(data_dim)

  results = []
  for model_params in models:
    results.append(eval_cv(n_splits, data, model_params))

  # process the results of the simple model-search & cross-validation
  results = extract_res_cross_validation(results, data)

  return results

def eval_models(data, model_family='classic'):
  if model_family == 'classic':
    res = eval_classic_ml_models(data, N_SPLITS)
  elif model_family == 'nn':
    res = eval_nn_models(data, N_SPLITS)

  return res