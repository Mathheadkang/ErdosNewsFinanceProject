import os
import pandas as pd
import sklearn as sk


def factor_model(config, logger):
	"""
	Factor model
	"""

	data = pd.read_csv(
		os.path.join(
			config['info']['local_data_path'],
			"data_clean",
			config['fin_preprocessing']['output']['stock_factor_data_file']
		)
		, index=False
	)


	# Split data into training and testing
	logger.info(f'Split data into training and testing. The test data is started from {config['info']['test_start_date']}')

	# fit the factor model on each stock
	logger.info('Start factor modeling on training data...')

	# predict the stock return on testing data
	logger.info('Start factor modeling on testing data...')

	# save the model

	# save the residuals