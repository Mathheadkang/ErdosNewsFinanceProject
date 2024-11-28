
import os
import ast
import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, HdpModel
from gensim.models.coherencemodel import CoherenceModel


def topic_model_driver(config, logger):
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
		if 'LDA' in config['news_model']['method']:
			model = LdaMulticore(
				corpus = corpus_train,
				id2word = dict,
				iterations = 100,
				num_topics = config['news_model']['lda_topics'],
				workers = config['news_model']['lda_num_cores'],
				passes = 100
			)
		elif 'HDP' in config['news_model']['method']:
			hdp_model = HdpModel(corpus = corpus_train, id2word = dict)
			model = hdp_model.suggested_lda_model()
		else:
			raise ValueError("The method is not supported.")
		
		# Compute the coherence score of topics
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
		corpus = [dict.doc2bow(doc) for doc in data["token"]]
		topics_inference = model.get_document_topics(corpus)
		data['topic'] = [sorted(model[corpus][text])[0][0] for text in range(len(topics_inference))]

		# Save the data
		data.to_csv(
			os.path.join(
				config['info']['local_data_path'],
				"model_news",
				config['news_model']['method'] + "_" + str(num_topics) + "_" + config['news_model']['output']['news_topic_file']
			),
			index = False
		)

		# Save the model
		model.save(
			os.path.join(
				config['info']['local_data_path'],
				"model_news",
				config['news_model']['method'] + "_" + str(num_topics) + "_" + config['news_model']['output']['model_file']
			)
		)

		# Save the topics
		topics = pd.DataFrame(
			{
				"topic": range(num_topics),
				"words": [model.show_topic(topicid, topn=20) for topicid in range(num_topics)]
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


		return model