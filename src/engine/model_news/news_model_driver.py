
import os
import ast
import pandas as pd
from joblib import Parallel, delayed

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, HdpModel
from gensim.models.coherencemodel import CoherenceModel


def topic_model_driver(config, logger):
		if len(config['news_model']['input']['news_data_file']) > 1:
			# Load the data
			dataset = config['news_model']['input']['news_data_file']
			df = []
			for file in dataset:
				df.append(pd.read_csv(
					os.path.join(
						config['info']['local_data_path'],
						"data_clean/All_news_years",
						file
					)
				))
			data = pd.concat(df, ignore_index=True)
		else:
			# Load the data
			data = pd.read_csv(
				os.path.join(
					config['info']['local_data_path'],
					"data_clean",
					config['news_model']['input']['news_data_file']
				)
			)

		# Select the columns
		data = data[["date", "token"]]
		data = data.sort_values(by="date")
		data = data.reset_index(drop=True)
		data['token'] = data['token'].apply(lambda row: [str(token) for token in ast.literal_eval(row)])

		logger.info(f"Loaded data in shape: {data.shape}")

		# train, validation, test data split
		data_train = data[
			(data["date"] >= config['data']['start_date']) &
			(data["date"] < config['data']['start_date_validation'])
		]

		# Training data
		dict = Dictionary(data_train["token"])
		dict.filter_extremes(
			no_below = config['news_model']['dict_no_below'],
			no_above = config['news_model']['dict_no_above'],
			keep_n = config['news_model']['dict_keep_n'])
		corpus_train = [dict.doc2bow(doc) for doc in data_train["token"]]

		# Train the model
		logger.info(f"Training the model with {config['news_model']['method']} method...")
		if 'LDA' in config['news_model']['method']:
			model = LdaMulticore(
				corpus = corpus_train,
				id2word = dict,
				iterations = 100,
				num_topics = config['news_model']['topics'],
				workers = config['news_model']['lda_num_cores'],
				passes = 100
			)
		elif 'HDP' in config['news_model']['method']:
			hdp_model = HdpModel(corpus = corpus_train, id2word = dict, T=config['news_model']['topics'])
			model = hdp_model.suggested_lda_model()
		else:
			raise ValueError("The method is not supported.")
		logger.info(f"Model training is done.")

		# Compute the coherence score of topics
		logger.info(f"Computing the coherence score of topics...")
		coherence_model = CoherenceModel(
			model = model,
			texts = data_train["token"],
			corpus = corpus_train,
			dictionary = dict,
			coherence = 'c_v')
		coherence_score = round(coherence_model.get_coherence(), 4)

		num_topics = model.num_topics
		logger.info(f"Coherence Score of {config['news_model']['method']} model with {num_topics} topics: {coherence_score}")
		

		# Predict the topics on all data
		logger.info(f"Predicting the topics on all data (training and testing)...")
		corpus = [dict.doc2bow(doc) for doc in data["token"]]
		topics_inference = model.get_document_topics(corpus)
		def sorted_topics(topics):
			return sorted(topics, key=lambda x: x[1], reverse=True)
		topics_inference_ = Parallel(n_jobs=-1)(delayed(sorted_topics)(topics) for topics in topics_inference)

		data['key_topic'] = [topics[0][0] for topics in topics_inference_]
		data['key_topic_wights'] = [topics[0][1] for topics in topics_inference_]
		data['all_topics'] = topics_inference_
		logger.info(f"Prediction is done.")

		# Save the data
		data_output = data[["date", "key_topic", 'key_topic_wights', 'all_topics']]
		data_output.to_csv(
			os.path.join(
				config['info']['local_data_path'],
				"model_news",
				config['news_model']['method'] + "_" + str(num_topics) + "_" + config['news_model']['output']['news_topic_file']
			),
			index = False
		)

		# Save the model
		model_path = os.path.join(
			config['info']['local_data_path'],
			"model_news",
			config['news_model']['method'] + "_" + str(num_topics) + "_" + config['news_model']['output']['model_file']
		)
		model.save(model_path)
		# model = HdpModel.load(model_path)

		# Save the topics
		topn = config['news_model']['topn']
		topic_words = [model.show_topic(topicid, topn=topn) for topicid in range(num_topics)]
		topic_words = [[
			[word for word, _ in topic],
			[round(wt, 6) for _, wt in topic]
		]	for topic in topic_words]
		topics = pd.DataFrame(
			{
				"topic": range(num_topics),
				"words": [topic[0] for topic in topic_words],
				"weights": [topic[1] for topic in topic_words]
			}
		)
		topics.to_csv(
			os.path.join(
				config['info']['local_data_path'],
				"model_news",
				config['news_model']['method'] + "_" + str(num_topics) + "_" + config['news_model']['output']['topic_file']
			),
			index = False
		)


		return None