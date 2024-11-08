
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
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy


# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the stopwords
stop_words = set(stopwords.words('english'))
# Local path
Local_path = "/Users/xiaokangwang/Documents/PycharmProjects/Projects for Erdos 2024 fall/data_set/all-the-news-2-1.csv"

#############################################################
# Drop columns
columns_to_drop = ['Unnamed: 0', 'author', 'year', 'month', 'day', 'url']

def drop_columns(df):
    drop = []
    for item in df.columns:
        if item in columns_to_drop:
            drop.append(item)
    return df.drop(drop,axis=1)

# Drop null articles:
subset = ['article']

def drop_null_articale(df):
    return df.dropna(subset=subset, inplace=True)


# Selecting the publishers
publisher = {}

def select_publishers(df):
    return df[df['publication'].isin(publisher)]

# Get the first 100 words of the article
def get_first_words(article_text):
    # Split the article text into words
    words = article_text.split()
    # Get the first 100 words
    first_100_words = words[:100]
    # Join them back into a string
    result = ' '.join(first_100_words)
    return result

# Get the tokenized article text without punctuations
def get_tokenized_words_with_no_punctuation(text):
    # Tokenize the text
    words = nltk.word_tokenize(text, language="english")
    # Remove the punctuations
    words_no_punctuation = [word.lower() for word in words if word.isalnum()]
    return words_no_punctuation

# Stop words removal
def remove_stop_words(words):
    words_no_stop_words = [word for word in words if word not in stop_words]
    return words_no_stop_words

# Lemmatization
def lemmatize_words(words):
    #words = get_tokenized_words_with_no_punctuation(article_text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

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


# Preprocessing pipeline:
def preprocess(df):
    # Drop columns
    df = drop_columns(df)   
    # Drop null articles
    drop_null_articale(df)
    
    # Select publishers
    #df = select_publishers(df)
   
    # Get the first 100 words of the article
    df['summary'] = df['article'].apply(get_first_words)
   
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






if __name__ == "__main__":
    #######################################################################
    # Time recoding
    start = time.time()

    ############################################################################################
    # This is for loading the example data set
    # df = pd.read_csv('data_set/Example_All_news.csv')
    # print(df.head())
    # df = preprocess(df)
    # time1 = time.time()
    # print("The overall time to load the data is ", time1-start, "seconds")
    # print(df.head())
    # df = pd.concat([df, df['token'].apply(lambda x: extract_ner_features(' '.join(x)))], axis=1)
    # time2 = time.time()
    # print("The overall time to extract NER is ", time2-time1, "seconds")
    # print(df.head())

    # df.to_csv('data_set/Preprocessed_Example_All_news.csv')

    # del df

    ############################################################################################
    # Define the chunk size
    chunksize = 10000
    i = 0
    time0 = start
    processed_chunks = []
    
    for chunk in pd.read_csv(Local_path, chunksize=chunksize):
        chunk = preprocess(chunk)
        time1 = time.time()
        print("Load the ",i ,"th chunk use time ", time1-time0, "seconds")
        
        ## Doing NER extraction will take a long time, for now, we will skip this step, later we will use multiprocessing to speed up the process
        # chunk = pd.concat([chunk, chunk['token'].apply(lambda x: extract_ner_features(' '.join(x)))], axis=1)
        # print("Extract NER for the ",i ,"th chunk use time ", time2-time1, "seconds")
        
        processed_chunks.append(chunk)

        time2 = time.time()

        print("Already uses time: ", time2-start, "seconds")

        i += 1
        time0 = time.time()
        del chunk
    
    if processed_chunks:
        dataset = pd.concat(processed_chunks)
        dataset.to_csv('data_set/Preprocessed_All_news.csv', index=False)



    end = time.time()
    print("To process", i, "chunks the overall time is ", end-start, "seconds")
