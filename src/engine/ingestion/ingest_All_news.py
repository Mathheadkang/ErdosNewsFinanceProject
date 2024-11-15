import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


#import clean.clean_All_news as cc
import pandas as pd
import time

# def preprocess_custom(df, columns_to_drop, subset, publishers):
#     df = cc.drop_columns(df, columns_to_drop)
#     # Drop rows where 'article' is NaN
#     cc.drop_null_article(df, subset)
#     # Select the publishers
#     df = cc.select_publishers(df, publishers)
#     # tokenize the article
#     df['token'] = df['article'].apply(cc.get_tokenized_words_with_no_punctuation)
#     # Get the word count
#     df['word_count'] = df['token'].apply(len)
#     # Filter the articles based on word count
#     df = df[(df['word_count'] > 100) & (df['word_count'] < 1000)]
#     # Drop the article column
#     df = df.drop('article', axis=1)
#     # Remove stop words
#     df['token'] = df['token'].apply(cc.remove_stop_words)
    
#     return df

def ingest_example(config, logger):
    # Return the local path
    local_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_file'])
    logger.info('Start processing the all_news data set...')
    # Load the first 10000 rows
    df = pd.read_csv(local_path, nrows=10000)
    
    # Preprocess the data
    save_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_eg1_file'])
    df.to_csv(save_path, index=False)
    logger.info('The example data set has been saved to {}'.format(save_path))

def ingest_k_example(config, logger):
    try:
        # Calculate rows to skip
        k = config['news_ingestion']['input']['k']
        skiprows = k * 10000
        
        # Get file path
        local_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_file'])
        
        # Log start

        logger.info(f'Reading chunk {k} from {local_path}')
        
        # Read specific chunk
        df = pd.read_csv(local_path, skiprows=skiprows, nrows=10000)
         
        # Preprocess the data
        save_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_egk_file'])
        df.to_csv(save_path, index=False)
        logger.info(f'The example data set has been saved to {save_path}')
              
    except Exception as e:
        if logger:
            logger.error(f'Error reading chunk {k}: {str(e)}')
        raise

# def ingest_EDA(config, logger):
#     EDA_chunks = []
#     i = 0
#     start = time.time()
#     t0 = start
#     try:
#         local_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_file'])
#         for chunk in pd.read_csv(local_path, chunksize=10000):
#             # Preprocess the chunk
#             processed_chunk_EDA = preprocess_custom(chunk, config['news_ingestion']['input']['columns_to_drop'], config['news_ingestion']['input']['subsets_dropna'], config['news_ingestion']['input']['publisher'])[['date','section','word_count']]
#             EDA_chunks.append(processed_chunk_EDA)
#             t1 = time.time()
#             logger.info(f'Processed {i} chunks in {t1 - t0} seconds')
#             t0 = t1
#             i += 1
#     except Exception as e:
#         if logger:
#             logger.error(f'Error processing EDA data: {str(e)}')
#         raise
#     # Concatenate the processed chunks
#     processed_EDA = pd.concat(EDA_chunks)
#     save_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['eda_all_news_file'])
#     processed_EDA.to_csv(save_path, index=False)
#     end = time.time()
#     logger.info(f'EDA data saved to {save_path}. Total time: {end - start} seconds')


def ingest_politics(config, logger):
    politics_chunks = []
    i = 0
    start = time.time()
    t0 = start
    try:
        local_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_file'])
        for chunk in pd.read_csv(local_path, chunksize=10000):
            chunk = chunk.dropna(subset=['section'])
            filtered_chunk = chunk[chunk['section'] == 'Politics']
            politics_chunks.append(filtered_chunk)
            t1 = time.time()
            logger.info(f'Processed {i} chunks in {t1 - t0} seconds')
            t0 = t1
            i += 1
    except Exception as e:
        if logger:
            logger.error(f'Error processing politics data: {str(e)}')
        raise
    # Concatenate the processed chunks
    politics_dataset = pd.concat(politics_chunks)
    save_path = os.path.join(config['info']['local_data_path'], 'data_raw', config['news_ingestion']['input']['all_news_politic_file'])
    politics_dataset.to_csv(save_path, index=False)
    end = time.time()
    logger.info(f'Politics data saved to {save_path}. Total time: {end - start} seconds')