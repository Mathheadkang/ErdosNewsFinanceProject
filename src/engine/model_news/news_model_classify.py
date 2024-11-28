import os
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import keras


############################################

def get_tfidf_headline(config, logger):
    """
    Get the tfidf of the headline
    """
    # Load data
    load_path = os.path.join(config['info']['local_data_path'],'data_clean', config['news_preprocessing']['output']["news_head_line_cleaned"])
    headline = pd.read_csv(load_path)

    # rename the category
    # Set the category we want to predict
    categorys_to_predict = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'ELSE']

    # Rename the category
    headline['category'] = headline['category'].apply(lambda x: x if x in categorys_to_predict else 'ELSE')

   
    # Sampling the data
    sameple_size = 10000
    sampled = headline.groupby('category').apply(lambda x: x.sample(sameple_size, random_state=42)) 

    # Choose the words that appear in at least 5 documents and at most 50% of the documents
    logger.info('Fitting the tfidf vectorizer')
    
    tfidf = TfidfVectorizer(max_df=0.5, min_df=5)
    tfidf.fit(sampled['headline_summary_tokenized'])
    tfidf_matrix = tfidf.transform(sampled['headline_summary_tokenized'])

    logger.info('Tfidf vectorizer fitted')

    # Use the nonnegative matrix factorization to reduce the dimensionality of the tfidf matrix
    logger.info('Fitting the nmf model')
    nmf = NMF(n_components=50, max_iter=500)
    nmf.fit(tfidf.transform(sampled['headline_summary_tokenized']))
    logger.info('Nmf model fitted')

    # Save the tfidf and nmf
    tfidf_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_headline'])
    nmf_model_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_nmf_headline'])
    tfidf_matrix_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_nmf_matrix'])

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(nmf, nmf_model_path)
    joblib.dump(tfidf_matrix, tfidf_matrix_path)


    logger.info('Tfidf and nmf model saved')

    # Save the sampled data
    save_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['headline_sampled'])
    sampled.to_csv(save_path, index=False)
    logger.info('Sampled data saved')

################################################
class customized_voting_classifier:
    """
    This is the customized voting classifier
    """
    def __init__(self, X_train):
        self.log = LogisticRegression(C=1000, solver='newton-cg')
        self.rf = RandomForestClassifier(n_estimators=700, max_depth=200)
        self.xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth = 10, n_estimators = 500)
        self.cnn = keras.models.Sequential()
        # Add an additional dimension to the data for the convolutional layer
        self.cnn.add(keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))

        # Add the convolutional layer
        self.cnn.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))

        # Add the max pooling layer
        self.cnn.add(keras.layers.MaxPooling1D(pool_size=2))

        # Flatten the output
        self.cnn.add(keras.layers.Flatten())

        # Add the dense layers
        self.cnn.add(keras.layers.Dense(128, activation='relu'))
        self.cnn.add(keras.layers.Dropout(0.4))
        self.cnn.add(keras.layers.Dense(64, activation='relu'))
        self.cnn.add(keras.layers.Dropout(0.3))
        self.cnn.add(keras.layers.Dense(4, activation='softmax'))

        self.cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Lable encoder
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train):
        print("Start to train the models: Logistic")
        self.log.fit(X_train, y_train)
        print("Logistic Regression has been trained")
        print("Start to train the models: Random Forest")
        self.rf.fit(X_train, y_train)
        print("Random Forest has been trained")
        # Encode the labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        

        # One-hot encode the labels
        y_train_encoded_cnn = keras.utils.to_categorical(y_train_encoded, num_classes=4)
       
        print("Start to train the models: XGBoost")
        self.xgb.fit(X_train, y_train_encoded)
        print("XGBoost has been trained")
        # Early stopping
        print("Start to train the models: CNN")
        early_stopping_cnn = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True, min_delta=0.0001, verbose=1)
        self.cnn.fit(X_train, y_train_encoded_cnn, epochs=200, batch_size=500, callbacks=[early_stopping_cnn], verbose=1)
        print("CNN has been trained")

    def predict(self, X_test):
        log_prob = self.log.predict_proba(X_test)
        rf_prob = self.rf.predict_proba(X_test)
        xgb_prob = self.xgb.predict_proba(X_test)
        cnn_prob = self.cnn.predict(X_test)

        weight = [0.2, 0.2, 0.4, 0.2]

        average_prob = log_prob*weight[0] + rf_prob*weight[1] + xgb_prob*weight[2] + cnn_prob*weight[3]

        y_pred = self.label_encoder.inverse_transform(np.argmax(average_prob, axis=1))

        return y_pred

################################################

def fit_the_headline_classifier(config, logger):
    """
    Fit the headline classifier
    """
    tfidf_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_headline'])
    nmf_model_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_nmf_headline'])
    tfidf_matrix_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_nmf_matrix'])
    load_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['headline_sampled'])
    # Load the data
    try:   
        tfidf = joblib.load(tfidf_path)
        nmf = joblib.load(nmf_model_path)
        tfidf_matrix = joblib.load(tfidf_matrix_path)
        sampled = pd.read_csv(load_path)
        logger.info('Tfidf and nmf model loaded')

    except Exception as e:
        logger.warning(f'Error loading the tfidf and nmf model: {e}')
        logger.info('Start fitting the tfidf and nmf model')

        # Fit the model now
        get_tfidf_headline(config, logger)

    # Retry loading models after fitting
        try:
            tfidf = joblib.load(tfidf_path)
            nmf = joblib.load(nmf_model_path)
            tfidf_matrix = joblib.load(tfidf_matrix_path)
            sampled = pd.read_csv(load_path)
            logger.info('Tfidf and nmf model loaded after fitting')
        except Exception as e:
            logger.error(f'Error loading the tfidf and nmf model after fitting: {e}')
            raise SystemExit('Failed to load models after fitting. Exiting...')
        
    # Initialize the classifier
    logger.info('Start fitting the classifier')
    transformed_data = nmf.transform(tfidf.transform(sampled['headline_summary_tokenized']))
    logger.info('Input data transformed')
    classifier = customized_voting_classifier(transformed_data)
    # Fit the classifier
    logger.info('Start fitting the classifier')
    classifier.fit(transformed_data, sampled['category'])
    logger.info('Classifier fitted')

    # Save the classifier
    classifier_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['classification_headlines'])
    joblib.dump(classifier, classifier_path)

    logger.info(f'Classifier saved to {classifier_path}')


#####################################################

def predict_the_all_news(config, logger):
    """
    predict the all news data set
    """
    # Load the data
    load_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['news_preprocessing']['output']['all_news_cleaned_file'])
    classifier_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['classification_headlines'])
    tfidf_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_headline'])
    nmf_model_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_nmf_headline'])
    # Load the classifier
    try:
        classifier = joblib.load(classifier_path)
        logger.info('Classifier loaded')
    except Exception as e:
        logger.error(f'Error loading the classifier: {e}')
        raise SystemExit('Failed to load the classifier. Exiting...')
    # Load the tfidf and nmf model
    try:
        tfidf = joblib.load(tfidf_path)
        nmf = joblib.load(nmf_model_path)
        logger.info('Tfidf and nmf model loaded')
    except Exception as e:
        logger.error(f'Error loading the tfidf and nmf model: {e}')
        raise SystemExit('Failed to load the tfidf and nmf model. Exiting...')
    
    # Define the chunk size
    chunksize = 10000
    i = 0
    predicted_chunks = []

    # Start the timer
    start = time.time()
    t0 = start

    logger.info('Start predicting the all_news data set...')

    try:
        for chunk in pd.read_csv(load_path, chunksize=chunksize):
            logger.info(f'Predicting chunk {i}...')
            chunk.rename(columns={'section':'category'}, inplace=True)
            # Transform the data
            logger.info(f'Transforming the chunk {i}...')
            chunk['summary'] = chunk['token'].apply(lambda x: x.split()[2:102])
            transformed_data = nmf.transform(tfidf.transform(chunk['summary'].apply(lambda x: ' '.join(x))))
            logger.info(f'The chunk {i} transformed')
            # Predict the data
            logger.info(f'Predicting the chunk {i}...')
            chunk['predicted_category'] = classifier.predict(transformed_data)
            # Append the predicted chunk to the list
            predicted_chunks.append(chunk)
            i += 1
            logger.info(f'Chunk {i} predicted uses {time.time() - t0} seconds')
            t0 = time.time()
            # Delete the chunk
            del chunk
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
    except Exception as e:
        logger.error(f'Error processing the all_news data set: {e}')

    # Finish predicting
    tf = time.time()
    logger.info(f'All_news data set predicting completed.Used time: {tf - start} seconds')

    # Concatenate the predicted chunks
    logger.info('Concatenating the predicted chunks...')
    predicted_all_news = pd.concat(predicted_chunks)
    logger.info('Concatenation completed.')

    # Save the predicted data to the local file
    save_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['news_preprocessing']['output']['all_news_predicted_file'])
    predicted_all_news.to_csv(save_path, index=False)
    logger.info(f"Predicted data saved to {save_path}")

    end = time.time()
    logger.info(f'Predicting all_news data set completed. Total time: {end - start} seconds')


#####################################################
def predict_the_all_news_eg(config, logger):
    """
    predict the all news data set
    """
    # Load the data
    load_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['news_preprocessing']['output']['all_news_eg1_cleaned_file'])
    classifier_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['classification_headlines'])
    tfidf_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_headline'])
    nmf_model_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['input']['tfidf_nmf_headline'])
    # Load the classifier
    try:
        classifier = joblib.load(classifier_path)
        logger.info('Classifier loaded')
    except Exception as e:
        logger.error(f'Error loading the classifier: {e}')
        raise SystemExit('Failed to load the classifier. Exiting...')
    # Load the tfidf and nmf model
    try:
        tfidf = joblib.load(tfidf_path)
        nmf = joblib.load(nmf_model_path)
        logger.info('Tfidf and nmf model loaded')
    except Exception as e:
        logger.error(f'Error loading the tfidf and nmf model: {e}')
        raise SystemExit('Failed to load the tfidf and nmf model. Exiting...')
    

    # Start the timer
    start = time.time()
    

    logger.info('Start predicting the all_news data set...')

    try:
        df = pd.read_csv(load_path)
        logger.info(f'Predicting the csv...')
        df.rename(columns={'section':'category'}, inplace=True)
        # Transform the data
        logger.info(f'Transforming the csv...')
        df['summary'] = df['token'].apply(lambda x: x.split()[2:102])
        transformed_data = nmf.transform(tfidf.transform(df['summary'].apply(lambda x: ' '.join(x))))
        logger.info(f'The csv transformed')
        # Predict the data
        logger.info(f'Predicting the csv...')
        df['predicted_category'] = classifier.predict(transformed_data)
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
    except Exception as e:
        logger.error(f'Error processing the all_news data set: {e}')

    # Finish predicting
    tf = time.time()
    logger.info(f'All_news data set predicting completed.Used time: {tf - start} seconds')

   
    # Save the predicted data to the local file
    save_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['news_preprocessing']['output']['all_news_predicted_eg1_file'])
    df.to_csv(save_path, index=False)
    logger.info(f"Predicted data saved to {save_path}")

    end = time.time()
    logger.info(f'Predicting all_news data set completed. Total time: {end - start} seconds')


            





