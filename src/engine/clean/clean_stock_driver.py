
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
