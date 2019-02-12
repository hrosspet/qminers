from qminers.config import CONFIG
from qminers.utils import load_data, prepare_data_financial, prepare_data_financial_seasonal, prepare_data_financial_seasonal_economic, prepare_data, process_results
from qminers.evaluation import eval_models


if __name__ == "__main__":
    SEQUENCE_LENGTH = CONFIG['SEQUENCE_LENGTH']

    data = load_data()

    data_preparation_functions = [
      prepare_data_financial,
      prepare_data_financial_seasonal,
      prepare_data_financial_seasonal_economic,
    ]
    model_families = ['classic', 'nn']

    results = {}

    for data_preparation_function in data_preparation_functions:
        name = data_preparation_function.__name__
        results[name] = {}

        for model_family in model_families:
            print('Evaluating data processed with:', name, 'and model family:', model_family)
            current_data = prepare_data(data, data_preparation_function, model_family)
            res = eval_models(current_data['train_val'], model_family)
            results[name][model_family] = res

    process_results(results, data_preparation_functions)