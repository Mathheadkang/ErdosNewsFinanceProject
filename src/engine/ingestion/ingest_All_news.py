import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


#import clean.clean_All_news as cc
import pandas as pd
import time


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