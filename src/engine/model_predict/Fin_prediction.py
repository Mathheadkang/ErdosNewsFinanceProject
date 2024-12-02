import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler



##############################################
def data_merging(config, logger):
    """
    Merge the data of the topic of news and the financial data
    """
    # Load the topic news data
    news_path = os.path.join(config['info']['local_data_path'], 'model_news', config['news_model']['output']['news_topic_file'])
    news_topics = pd.read_csv(news_path)
    # Round the topic time to the date. We need to handle ISO 8601 date format
    news_topics['date'] = pd.to_datetime(news_topics['date'], format = 'mixed').apply(lambda x: x.date().isoformat())

    # Load the financial data
    stock_path = os.path.join(config['info']['local_data_path'], 'data_clean', config['fin_model']['output']['abnormal_return_file'])
    abreturn = pd.read_csv(stock_path)
    # Make sure the date of the same type
    abreturn['date'] = pd.to_datetime(abreturn['date'])

    logger.info('Data loaded successfully')

    # Select the ticker
    ticker = config['predict_model']['ticker']
    ab_ticker = abreturn[abreturn['ticker'] == ticker]

    logger.info(f"Find the ticker:{ticker}")

    # Select the big topics
    topics = news_topics['key_topic'].unique()
    threshold = config['predict_model']['topic_threshold']
    big_topics = [topic for topic in topics if news_topics[news_topics['key_topic'] == topic]['key_topic'].count() > threshold]
    # Filtered news topics data
    news_topics_filtered = news_topics[news_topics['key_topic'].isin(big_topics)]

    # Merge the data
    # Define the merged data DataFrame
    merged = pd.DataFrame()
    merged['date'] = ab_ticker['date']
    merged['return'] = ab_ticker['return']
    merged['return_hat_reduced'] = ab_ticker['return_hat']

    # Group by date and key_topic, then count occurrences
    topic_counts = news_topics_filtered.groupby(['date', 'key_topic']).size().unstack(fill_value=0)

    # Ensure the 'date' column in topic_counts is of the same type as in merged
    topic_counts.index = pd.to_datetime(topic_counts.index)

    # Merge the topic counts with the merged DataFrame
    merged = merged.merge(topic_counts, on='date', how='left').fillna(0)

    # Save the merged data
    merged_path = os.path.join(config['info']['local_data_path'], 'model_pre_train', config['predict_model']['merged_data'])
    merged.to_csv(merged_path, index=False)

    logger.info(f'Data merged successfully, saved to {merged_path}')
    


##############################################
def predicting_price(config, logger):
    """
    Predict the price of the stock
    """

    # Load the merged data
    try:
        merged_path = os.path.join(config['info']['local_data_path'], 'model_pre_train', config['predict_model']['merged_data'])
        merged = pd.read_csv(merged_path)
        logger.info('Data loaded successfully')
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        logger.info('Try to merge the data again')
        data_merging(config, logger)
    try:
        merged = pd.read_csv(merged_path)
        logger.info('Data loaded successfully')
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise SystemExit('Failed to load models after merging. Exiting...')
        

    #Combine the data with the lagged information
    # Use the lag of 2 features

    # Shift the data by 1
    merged_shift = merged.copy()
    merged_shift_1 = merged_shift.shift(periods=1)

    merged_shift_1.drop(columns = ['date', 'return', 'return_hat_reduced'], inplace=True)
    merged_shift_1.columns = [f"{col}_lag{1}" for col in merged_shift_1.columns]

    # Shift the data by 2
    merged_shift_2 = merged_shift.shift(periods=2)

    merged_shift_2.drop(columns = ['date', 'return', 'return_hat_reduced'], inplace=True)
    merged_shift_2.columns = [f"{col}_lag{2}" for col in merged_shift_2.columns]


    # Concatenate the data

    merged_concate = pd.concat([merged_shift, merged_shift_1], axis=1)
    merged_concate = pd.concat([merged_concate, merged_shift_2], axis=1)
    
    merged_concate.drop(merged_concate.index[0], inplace=True)
    merged_concate.drop(merged_concate.index[0], inplace=True)
   
    merged_concate.columns = merged_concate.columns.astype(str)

    # Preprocess the data
    # Split the data into train and test

    scale = StandardScaler()
    train_data = merged_concate[merged_concate['date'] <= '2018-12-31']
    test_data = merged_concate[merged_concate['date'] > '2018-12-31']

    X_train = train_data.drop(columns=['date', 'return','return_hat_reduced'])
    X_train_scaled = scale.fit_transform(X_train)
    y_train = train_data['return'] - train_data['return_hat_reduced']
    X_test = test_data.drop(columns=['date', 'return', 'return_hat_reduced'])
    X_test_scaled = scale.transform(X_test)
    y_test = test_data['return'] - test_data['return_hat_reduced']

    logger.info('Data preprocessed successfully')
    
    # Predict the price using XGBoost
    logger.info('Predicting the price using XGBoost')
    n_estimators = config['predict_model']['n_estimators']
    max_depth = config['predict_model']['max_depth']
    reg_lambda = config['predict_model']['reg_lambda']
    xgb_model = xgb.XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, objective='reg:squarederror', random_state=42, reg_lambda=reg_lambda)
    xgb_model.fit(X_train_scaled, y_train)

    logger.info('Model trained successfully')

    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_full = y_pred + test_data['return_hat_reduced']
    logger.info('Prediction completed successfully')

    # Save the prediction
    pred = pd.DataFrame()
    pred['date'] = test_data['date']
    pred['return'] = test_data['return']
    pred['return_reduced'] = test_data['return_hat_reduced']
    pred['return_full'] = y_pred_full

    pred_path = os.path.join(config['info']['local_data_path'], 'model_pre_train', config['predict_model']['predict_price'])

    pred.to_csv(pred_path, index=False)
    logger.info(f'Prediction saved to {pred_path}')

