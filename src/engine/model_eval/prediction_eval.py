import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


##############################################
def measure_performance(y_true, y_pred_reduced, y_pred_full):
    mse_1 = mean_squared_error(y_true, y_pred_reduced)
    mae_1 = mean_absolute_error(y_true, y_pred_reduced)
    r2_1 = r2_score(y_true, y_pred_reduced)

    mse_2 = mean_squared_error(y_true, y_pred_full)
    mae_2 = mean_absolute_error(y_true, y_pred_full)
    r2_2 = r2_score(y_true, y_pred_full)
 
    
    return mse_1, mae_1, r2_1, mse_2, mae_2, r2_2

##############################################
def get_accuracy(y_true, y_pred_reduced, y_pred_full):
    y_true[y_true >0] = 1
    y_true[y_true <0] = -1

    y_pred_reduced[y_pred_reduced >0] = 1
    y_pred_reduced[y_pred_reduced <0] = -1

    y_pred_full[y_pred_full >0] = 1
    y_pred_full[y_pred_full <0] = -1

    accuracy_reduced = (y_true * y_pred_reduced > 0).mean()
    accuracy_full = (y_true * y_pred_full > 0).mean()

    return accuracy_reduced, accuracy_full

##############################################
def evaluation(config, logger):
    # Load the predicted data
    try:
        pred_path = os.path.join(config['info']['local_data_path'], 'model_pre_train', config['predict_model']['predict_price'])
        pred = pd.read_csv(pred_path)
        logger.info('Data loaded successfully')
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise SystemExit('Failed to load models after merging. Exiting...')
    
    baseline_mse = mean_squared_error(pred['return'], np.ones(pred['return'].shape)*pred['return'].iloc[0])
    mse_1, mae_1, r2_1, mse_2, mae_2, r2_2 = measure_performance(pred['return'], pred['return_reduced'], pred['return_full'])
    accuracy_reduced, accuracy_full = get_accuracy(pred['return'], pred['return_reduced'], pred['return_full'])

    logger.info(f'Model performance: \n'
                f'MSE reduced: {round(mse_1,6)}, MAE reduced: {round(mae_1, 6)}, R2 reduced: {round(r2_1, 6)}\n'
                f'MSE full: {round(mse_2,6)}, MAE full: {round(mae_2,6)}, R2 full: {round(r2_2,6)}\n'
                f'Accuracy reduced: {round(accuracy_reduced,6)}, Accuracy full: {round(accuracy_full,6)}')
    logger.info(f'Baseline MSE: {round(baseline_mse, 6)}')