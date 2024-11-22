import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

def factor_model(config, logger):
	"""
	Factor model
	"""


	# Load data
	data = pd.read_csv(
		os.path.join(
			config['info']['local_data_path'],
			"data_clean",
			config['fin_model']['input']['stock_factor_data']
		)
	)
	data['return_hat'] = 0.0
	data['residual'] = 0.0


	# Split data into training and testing
	logger.info(f'Split training data from {config['data']['start_date']} to {config['data']['start_date_validation']}')
	data_train = data[data['date'] < config['data']['start_date_validation']]

	logger.info(f'Split validation data from {config['data']['start_date_validation']} to {config['data']['start_date_test']}')
	data_val = data[(data['date'] >= config['data']['start_date_validation']) & (data['date'] < config['data']['start_date_test'])]	

	logger.info(f'Split testing data from {config['data']['start_date_test']} to {config['data']['end_date']}')
	data_test = data[data['date'] >= config['data']['start_date_test']]


	# fit the factor model on each stock
	logger.info('Start factor modeling on training data, and predicting test data...')

	tickers = data_train['ticker'].unique()
	X_variable = ["MKT", "SMB", "HML", "RMW", "CMA"]
	# scale the factors by dividing 100
	data_train[X_variable] = data_train[X_variable] / 100
	fitted_models = {}
	for ticker in tickers:
		# fit a factor model
		ticker_index_train = data_train['ticker'] == ticker
		X_train = data_train[ticker_index_train][X_variable]
		y_train = data_train[ticker_index_train]['return']
		fitted_models[ticker] = LinearRegression().fit(X_train, y_train)

		# save the model
		file_name = os.path.join(config['info']['local_data_path'], 'model_fin', f'{ticker}.joblib')
		joblib.dump(fitted_models[ticker], file_name)

		logger.info(f'The factor model for {ticker} is saved at {file_name}')

		# save the residuals
		y_hat_train = fitted_models[ticker].predict(X_train)
		data_train.loc[ticker_index_train, 'return_hat'] = y_hat_train
		data_train.loc[ticker_index_train, 'residual'] = y_train.values - y_hat_train

		ticker_index_val = data_val['ticker'] == ticker
		y_val = data_val[ticker_index_val]['return']
		y_hat_val = fitted_models[ticker].predict(data_val.loc[ticker_index_val][X_variable])
		data_val.loc[ticker_index_val, 'return_hat'] = y_hat_val
		data_val.loc[ticker_index_val, 'residual'] = y_val.values - y_hat_val

		ticker_index_test = data_test['ticker'] == ticker
		y_test = data_test[ticker_index_test]['return']
		y_hat_test = fitted_models[ticker].predict(data_test.loc[ticker_index_test][X_variable])
		data_test.loc[ticker_index_test, 'return_hat'] = y_hat_test
		data_test.loc[ticker_index_test, 'residual'] = y_test.values - y_hat_test

		logger.info(f'MSE for {ticker} on training data: {round(mse(y_train, y_hat_train), 6)}')
		logger.info(f'MSE for {ticker} on validation data: {round(mse(y_val, y_hat_val), 6)}')
		logger.info(f'MSE for {ticker} on testing data: {round(mse(y_test, y_hat_test), 6)}')

	# combine and save residuals
	data = pd.concat([data_train, data_val, data_test], axis=0)
	data.to_csv(
		os.path.join(
			config['info']['local_data_path'],
			'data_clean',
			config['fin_model']['output']['abnormal_return_file']
		),
		index=False
	)

	return None


def mse(y_true, y_pred):
	"""
	Mean squared error
	"""
	return np.mean((y_true - y_pred) ** 2)