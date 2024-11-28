#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the main function of all drivers to preidct stock with news.
Should use config/predict_stock_w_news.toml to configure the parameters.
"""

import os
import argparse
import errno
import datetime

from utils.config_tool import parse_config, save_config_copy
from utils.directory_tool import ensure_dir, get_directory_names
from utils.logging_tool import initialize_logger

# modules for finance data
from engine.ingestion.ingest_fin_driver import ingest_stock, ingest_factors
from engine.clean.clean_stock_driver import get_return, combine_stock_factors
from engine.model_fin.fin_model_driver import factor_model

# modules for news data
from engine.ingestion import ingest_All_news as ia
from engine.clean.clean_All_news import preprocess_all_news_main
from engine.model_news.news_model_classify import predict_the_all_news

# modules for predictive model

############################################
def main(opt_params):
		"""
		The main function to predict stock with news.

		Args:
				opt_params: Optional parameters via argparse

		Returns:
				None
		"""


		# Optional parameters
		config_filename = opt_params.config_filename
		dir_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


		# Configuration
		# Import the configuration file
		config_file = os.path.join(dir_project, config_filename)
		if os.path.exists(config_file):
			config = parse_config(config_file)
		else:
			# Raise an error if the configuration file does not exist
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
		
		# Add today to the configuration
		config['date']['today'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


		# Data directories
		# Forming data directories in the local data path
		dirs = get_directory_names(
			path_core_data = config["info"]["local_data_path"],
			dirs_names = config["info"]["dirs_names"]
		)


		# Save the config as a copy
		ensure_dir(dirs["config_archive"])
		save_config_copy(
			config_path = dirs["config_archive"],
			config = config,
			file_name = "predict_stock_w_news_copy_.toml"
		)


		# Logging
		# Initialize the logger
		ensure_dir(dirs["logs"])
		log_file = "{}_log_{}.txt".format(
			os.path.splitext(os.path.basename(config_filename))[0],
			config["date"]['today'])
		logger = initialize_logger(
			log_path = dirs["logs"],
			log_file = log_file
		)




		############################################
		# Starting pipeline

		# For finance data
		# Ingest finance data
		if config['pipeline']['fin_ingestion']:
			logger.info('==> Start ingesting finance data...')
			ensure_dir(dirs["data_raw"])
			ingest_stock(config, logger)
			ingest_factors(config, logger)
			logger.info('Finance data ingestion completed.')

		# finance preprocessing
		if config['pipeline']['fin_processing']:
			logger.info('==> Start processing finance data...')
			ensure_dir(dirs["data_clean"])
			get_return(config, logger)
			combine_stock_factors(config, logger)
			logger.info('Finance data processing completed.')

		# finance model
		if config['pipeline']['fin_model']:
			logger.info('==> Start modeling on finance data...')
			ensure_dir(dirs["model_fin"])
			factor_model(config, logger)
			logger.info('Finance modeling completed.')
		

		# For news data
		# news ingestion
		if config['pipeline']['news_ingestion']:
			logger.info('==> Start ingesting News data...')
			ensure_dir(dirs["data_raw"])
			ia.ingest_example(config, logger)
			ia.ingest_k_example(config, logger)
			ia.ingest_politics(config, logger)
			logger.info('News data ingestion completed.')
		
		# news preprocessing
		if config['pipeline']['news_processing']:
			logger.info('==> Start processing News data...')
			ensure_dir(dirs["data_clean"])
			preprocess_all_news_main(config, logger)
			logger.info('News data processing completed.')
		
		
		# news model
		if config['pipeline']['news_model']:
			logger.info('==> Start modeling on News data...')
			ensure_dir(dirs["model_news"])
			predict_the_all_news(config, logger)
			logger.info('News modeling completed.')


		# For predictive model
		# prediction
		if config['pipeline']['predict_model']:
			logger.info('==> Start training the predictive model...')
			ensure_dir(dirs["model_pre_train"])
			# --> to add a function here
			logger.info('Predictive model training completed.')


		# evaluation
		if config['pipeline']['predict_evaluation']:
			logger.info('==> Start evaluating the predictive model...')
			ensure_dir(dirs["model_pre_eval"])
			# --> to add a function here
			logger.info('Predictive model evaluation completed.')



############################################
if __name__ == '__main__':
		# Argument parsing
		parser = argparse.ArgumentParser(description='Predict stock with news.')
		parser.add_argument(
			'--config',
			type=str,
			default='config/predict_stock_w_news.toml',
			dest='config_filename',
			help='the path to the configuration file'
		)

		# Parse the arguments
		args = parser.parse_args()

		# Run the main function
		main(args)
