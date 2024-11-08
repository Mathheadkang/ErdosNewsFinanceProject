from yahoo_fin import stock_info as si
import pandas as pd
import os

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
					start_date = config["fin_ingestion"]["input"]["start_date"],
					end_date = config["fin_ingestion"]["input"]["end_date"],
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
