# This script is used to preprocess the all_news data set. It contains the following functions:



# drop_columns(df): This function drops the columns that are not needed for the analysis.

# drop_null_articale(df): This function drops the rows where the 'article' column is NaN.

# select_publishers(df): This function filters the data based on the publishers.

# get_first_words(article_text): This function extracts the first 100 words from the article text.

# get_tokenized_words_with_no_punctuation(text): This function tokenizes the text and removes the punctuations.

# remove_stop_words(words): This function removes the stop words from the tokenized words.

# lemmatize_words(words): This function lemmatizes the words.

# extract_ner_features(text): This function extracts the named entity recognition features from the text.

# preprocess(df): This function applies the preprocessing pipeline to the data frame.


#############################################################
import os

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import time


# Load the spaCy model
# check if the model is installed, if not install it in python
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

nlp = spacy.load('en_core_web_sm')

# Load the stopwords
stop_words = set(stopwords.words('english'))


#######################################################
# Drop columns

# (Need to specify the columns to drop in the config .toml file)

def drop_columns(df, columns_to_drop):
    drop = []
    for item in df.columns:
        if item in columns_to_drop:
            drop.append(item)
    return df.drop(drop,axis=1)

#######################################################
# Drop null articles: 

# (Need to specify the subset in the config .toml file)
#subset = ['article']

def drop_null_article(df, subset):
    return df.dropna(subset=subset, inplace=True)

#######################################################
# Selecting the publishers

# (Need to specify the publisher in the config .toml file)

def select_publishers(df, publisher):
    return df[df['publication'].isin(publisher)]

#######################################################
# Get the first x words of the article:

# (Need to specify the number of words in the config .toml file)

def get_first_words(article_text, num_words):
    # Split the article text into words
    words = article_text.split()
    # Get the first x words
    first_x_words = words[:num_words]
    # Join them back into a string
    result = ' '.join(first_x_words)
    return result


#######################################################
# Get the tokenized article text without punctuations
def get_tokenized_words_with_no_punctuation(text):
    # Tokenize the text
    words = nltk.word_tokenize(text, language="english")
    # Remove the punctuations
    words_no_punctuation = [word.lower() for word in words if word.isalnum()]
    return words_no_punctuation

#######################################################
# Stop words removal
def remove_stop_words(words):
    words_no_stop_words = [word for word in words if word not in stop_words]
    return words_no_stop_words

#######################################################
# Lemmatization
def lemmatize_words(words):
    #words = get_tokenized_words_with_no_punctuation(article_text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words


#######################################################
# Named entity recognition
def extract_ner_features(text):
    doc = nlp(text)
    entity_counts = {
        "PERSON": 0,
        "ORG": 0,
        "GPE": 0,
        "EVENT": 0,
        "PRODUCT": 0
    }
    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1
    return pd.Series(entity_counts)

#######################################################
# Preprocessing all_news pipeline:
def preprocess_all_news(df, columns_to_drop, subset, publisher, num_words):
    # Drop columns
    df = drop_columns(df, columns_to_drop)   
    # Drop null articles
    drop_null_article(df, subset)
    
    # Select publishers
    df = select_publishers(df, publisher)
   
    # Get the first x words of the article
    df['summary'] = df['article'].apply(lambda x: get_first_words(x, num_words))
  
    # Get the tokenized article text without punctuations
    df['token'] = df['article'].apply(get_tokenized_words_with_no_punctuation)
   
    # Get the word count
    df['word_count'] = df['token'].apply(len)
    
    # Filter the articles based on word count
    df = df[(df['word_count'] > 100) & (df['word_count'] < 1000)]
    # Drop the article column
    df = df.drop('article', axis=1)
  
    # Remove stop words
    df['token'] = df['token'].apply(remove_stop_words)

    # Lemmatization
    df['token'] = df['token'].apply(lemmatize_words)
    return df

############################################################################################################
# The main function to preprocess the all_news data set

def preprocess_all_news_main(config, logger):
    columns_to_drop = config['news_ingestion']['input']['columns_to_drop']
    subset = config['news_ingestion']['input']['subsets_dropna']
    publisher = config['news_ingestion']['input']['publisher']
    num_words = config['news_ingestion']['input']['num_words']
    
    #Local path
    local_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_file'])
    # Define the chunk size
    chunksize = 10000
    i = 0
    processed_chunks = []
    
    # Start the timer
    start = time.time()
    logger.info('Start processing the all_news data set...')
    t_0 = start

    try:
        for chunk in pd.read_csv(local_path, chunksize=chunksize):
            # Preprocess the chunk
            chunk = preprocess_all_news(chunk, columns_to_drop, subset, publisher, num_words)
            # Append the processed chunk to the list
            processed_chunks.append(chunk)
            i += 1
            # Print the progress
            t_1 = time.time()
            logger.info(f'Processed {i} chunks in {t_1 - t_0} seconds')
            t_0 = t_1
            # delete the chunk
            del chunk
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
    except Exception as e:
        logger.error(f'Error processing the all_news data set: {e}')

    # Finish preocessing
    t_f = time.time()
    logger.info(f'All_news data set processing completed.Used time: {t_f - start} seconds')

    # Concatenate the processed chunks
    logger.info('Concatenating the processed chunks...')
    processed_all_news = pd.concat(processed_chunks)
    logger.info('Concatenation completed.')

    # Save the processed data to the local file
    logger.info('Saving the processed data to the local file...')
    save_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['news_preprocessing']['output']['all_news_cleaned_file'])
    processed_all_news.to_csv(save_path, index=False)
    logger.info(f'Data saved successfully in {save_path}.')
    end = time.time()
    logger.info(f'Preprocessing all_news data set completed. Total time: {end - start} seconds')


    ############################################################################################################
def preprocess_all_news_eg1_main(config, logger):
    columns_to_drop = config['news_ingestion']['input']['columns_to_drop']
    subset = config['news_ingestion']['input']['subsets_dropna']
    publisher = config['news_ingestion']['input']['publisher']
    num_words = config['news_ingestion']['input']['num_words']
    
    #Local path
    local_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_eg1_file'])
    # Define the chunk size
    chunksize = 10000
    i = 0
    processed_chunks = []
    # Start the timer
    start = time.time()
    logger.info('Start processing the all_news data set...')
    t_0 = start

    try:
        for chunk in pd.read_csv(local_path, chunksize=chunksize):            
            # Preprocess the chunk
            chunk = preprocess_all_news(chunk, columns_to_drop, subset, publisher, num_words)
            # Append the processed chunk to the list
            processed_chunks.append(chunk)
            i += 1
            # Print the progress
            t_1 = time.time()
            logger.info(f'Processed {i} chunks in {t_1 - t_0} seconds')
            t_0 = t_1
            # delete the chunk
            del chunk
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
    except Exception as e:
        logger.error(f'Error processing the all_news data set: {e}')

    # Finish preocessing
    t_f = time.time()
    logger.info(f'All_news data set processing completed.Used time: {t_f - start} seconds')

    # Concatenate the processed chunks
    logger.info('Concatenating the processed chunks...')
    processed_all_news = pd.concat(processed_chunks)
    logger.info('Concatenation completed.')

    # Save the processed data to the local file
    logger.info('Saving the processed data to the local file...')
    save_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['news_preprocessing']['output']['all_news_eg1_cleaned_file'])
    processed_all_news.to_csv(save_path, index=False)
    logger.info(f'Data saved successfully in {save_path}.')
    end = time.time()
    logger.info(f'Preprocessing all_news data set completed. Total time: {end - start} seconds')