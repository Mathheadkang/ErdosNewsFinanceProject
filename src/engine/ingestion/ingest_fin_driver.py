from yahoo_fin import stock_info as si
import pandas as pd
import os

import pandas_datareader.data as pdr

def ingest_stock(config, logger):
		"""
		This function is used to ingest stock data from Yahoo Finance.
		"""
		# amazon_weekly= si.get_data("amzn", start_date="12/04/2009", end_date="12/04/2019", index_as_date = True, interval="1wk")

		# read the stock symbols from the local file
		symbol_list = pd.read_csv(
			os.path.join(
				config["info"]["local_data_path"],
				"data_raw",
				config["fin_ingestion"]["input"]["stock_symbol_file"]
			)
		)

		# load the list of tickers
		#ticker_list = ["amzn", "aapl", "ba"]
		ticker_list = symbol_list["Symbol"].tolist()

		# download the historical data of all the tickers
		logger.info("Downloading historical data for the following tickers: {}".format(ticker_list))

		historical_datas = {}
		for ticker in ticker_list:
				historical_datas[ticker] = si.get_data(
					ticker = ticker,
					start_date = config["data"]["start_date"],
					end_date = config["data"]["end_date"],
					interval = "1d")
		logger.info("Download completed.")

		historical_datas = pd.concat(historical_datas)
		# save the historical data to the local file
		historical_datas.to_csv(
			os.path.join(
				config["info"]["local_data_path"],
				"data_raw",
				config["fin_ingestion"]["output"]["stock_data_file"]
			)
		)


def ingest_factors(config, logger):
	"""
	This function is used to ingest Fama French factors from Ken French's website.
	"""

	# download the Fama French factors
	logger.info("Downloading Fama French factors...")
	factor_file = config["fin_ingestion"]["input"]["factor_file"]
	df_factors = pdr.DataReader(
		factor_file, 
		'famafrench', 
		start = config["data"]["start_date"],
		end = config["data"]["end_date"])[0]
	df_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)

	# log the download completion and count the number of rows
	logger.info(f"Fama-French data download completed. (from {df_factors.index[0]} to {df_factors.index[-1]}, range: {len(df_factors)} trading days)")

	# save the Fama French factors to the local file
	df_factors.to_csv(
		os.path.join(
			config["info"]["local_data_path"],
			"data_raw",
			config["fin_ingestion"]["output"]["factor_data_file"]
		)
	)