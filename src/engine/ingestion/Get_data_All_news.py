
#######################################################################

# This script reads the 'all-the-news-2-1.csv' file in chunks and filters the data for the 'Politics' section. Also, it
# provides an EDA data set and a k example data set. The filtered data is saved to a CSV file.


#######################################################################

import sys
sys.path.append('../clean')

import News_data_preprocessing as ndp
import pandas as pd
import time

start = time.time()

# Local path to the CSV file
Local_path = '/Users/xiaokangwang/Documents/PycharmProjects/Projects for Erdos 2024 fall/data_set/all-the-news-2-1.csv'

# Initialize an empty list to store the filtered chunks
filtered_chunks = []

# Initialize an emnpty list to store the chunks for EDA data set
EDA_chunks = []


# Customized-preprocessing pipeline, we do not need to lemmatize the words since it is time consuming:
def preprocess_custom(df):
    df = ndp.drop_columns(df)
    # Drop rows where 'article' is NaN
    ndp.drop_null_articale(df)
    # tokenize the article
    df['token'] = df['article'].apply(ndp.get_tokenized_words_with_no_punctuation)
    # Get the word count
    df['word_count'] = df['token'].apply(len)
    # Filter the articles based on word count
    df = df[(df['word_count'] > 100) & (df['word_count'] < 1000)]
    # Drop the article column
    df = df.drop('article', axis=1)
    # Remove stop words
    df['token'] = df['token'].apply(ndp.remove_stop_words)
    
    return df

# Define the chunk size
chunksize = 10000
i = 0
time0 = start

# Get the k th chunk
k = 50

# Read the CSV file in chunks
try:
    for chunk in pd.read_csv(Local_path, chunksize=chunksize):
        #######################################################################
        # For k example data set
        if i == k:
            chunk.to_csv('data_set/kExample_All_news.csv')
        
        #######################################################################    
        # # For EDA data set
        chunk_EDA = chunk.copy(deep=True)
        print(chunk_EDA.head(1))

        propcessed_chunk_EDA = preprocess_custom(chunk_EDA)[['date','section','word_count']]
        print(propcessed_chunk_EDA.head(1))      
        EDA_chunks.append(propcessed_chunk_EDA)

        #######################################################################
        # For politics data set
        print(chunk.head(1))
        # Drop rows where 'section' is NaN
        chunk = chunk.dropna(subset=['section'])
        
        # Filter the chunk for 'Politics' section
        filtered_chunk = chunk[chunk['section'] == 'Politics']
        
        # Append the filtered chunk to the list
        filtered_chunks.append(filtered_chunk)
        i += 1
        time1 = time.time()
        print("Process the ",i ,"th chunk use time ", time1-time0, "seconds")
        time0 = time1

        del chunk

except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Concatenate all the filtered chunks
if filtered_chunks:
    dataset_politics = pd.concat(filtered_chunks)
    
    # Save the concatenated DataFrame to a CSV file
    dataset_politics.to_csv('data_set/All_news_politics.csv', index=False)

    cleaned_dataset = ndp.preprocess(dataset_politics)
    cleaned_dataset.to_csv('data_set/Preprocessed_All_news_politics.csv', index=False)
else:
    print("No data was processed.")

if EDA_chunks:
    EDA_dataset = pd.concat(EDA_chunks)
    EDA_dataset.to_csv('data_set/EDA_All_news.csv', index=False)
else:
    print("No data was processed for EDA.")

end = time.time()

print("The overall time to load the data is ", end-start, "seconds")
print("There are in total ", i, "chunks")