
import pandas as pd
import os

def get_return(config, logger):
		"""
		This function is used to clean the stock data.
		"""
		# read the stock data from the local file
		stock_data = pd.read_csv(
			os.path.join(
				config["info"]["local_data_path"],
				"data_raw",
				config["fin_preprocessing"]["input"]["stock_data_file"]
			)
		)

		# calculate the returns of each stock
		stock_data["return"] = stock_data["close"].diff(periods=1) / stock_data["close"].shift(periods=1)
		
		# drop the the first row for each stock
		stock_data = stock_data.groupby("ticker").apply(lambda x: x.iloc[1:])
		stock_data.reset_index(drop=True, inplace=True)
		
		# rename the first two columns only

		stock_data.columns = ["stock", "date"] + list(stock_data.columns[2:])

		# column selection
		stock_data = stock_data[["date", "ticker", "return"]]

		# save the cleaned data to the local file
		stock_data.to_csv(
			os.path.join(
				config["info"]["local_data_path"],
				"data_clean",
				config["fin_preprocessing"]["output"]["stock_data_cleaned_file"]
			)
		)
		return None

def combine_stock_factors(config, logger):
	"""
	Combine stock and factors data
	"""

	stock = pd.read_csv(
		os.path.join(
			config['info']['local_data_path'],
			"data_clean",
			config['fin_model']['input']['stock_data']
		)
	)
	factors = pd.read_csv(
		os.path.join(
			config['info']['local_data_path'],
			"data_raw",
			config['fin_model']['input']['factor_data']
		)
	)
	stock['date'] = pd.to_datetime(stock['date'])
	factors['date'] = pd.to_datetime(factors['Date'])
	stock_factors = pd.merge(stock, factors, on='date', how='inner')
	# drop Date column
	stock_factors.drop('Date', axis=1, inplace=True)
	stock_factors.drop('Unnamed: 0', axis=1, inplace=True)

	stock_factors.to_csv(
		os.path.join(
			config['info']['local_data_path'],
			"data_clean",
			config['fin_preprocessing']['output']['stock_factor_data_file']
		)
		, index=False
	)

	return None
