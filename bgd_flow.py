#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-07-02 15:36:11 Tuesday

@author: Nikhil Kapila
"""

import os, logging, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from classes.ElecDatasetClass import ElecDataset
from classes.LSTMModelClass import LSTMModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from metaflow import FlowSpec, step, Parameter, card, catch
from metaflow.client import Flow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METAFLOW_SERVICE_URL = 'http://localhost:3000/'

class BGD_LinearFlow(FlowSpec):
    # Relative path from folder
    # file_path = Parameter('filepath', help='Path to the input dataframe.',
        # default='/Users/nikhilkapila/Developer/lstm-training/datasets/building_genome_dataset/data_subsets/bgd_Education_1001_5000.csv')
    cwd = os.getcwd()
    file_path = Parameter('filepath', help='Path to the input dataframe.', default='/datasets/building_genome_dataset/data_subsets/bgd_Education_1001_5000.csv')
    use_hourly = Parameter('use_hourly', help='Flag to toggle hourly forecasts.', default=True)
    hidden_units_lstm = Parameter('hidden_units_lstm', help='Define hidden units to LSTM.', default=20)
    lookback = Parameter('lookback',help='Amount of lag features to train on.', default=30)
    epochs = Parameter('epochs', help='Number of epochs to train on', default=50)
    model_type = Parameter('model_type', help='Forecasting model to use.', default='LSTM')
    dir = Parameter('dir', help='Save in directory.', default='metaflow_models/')
    # flow_tag = Parameter('flow_tag', help='Tag name for experiment.')

    @step
    def start(self):
        logger.info('Starting BGD linear flow..')
        # current.tags.add(self.flow_tag)
        self.model_name = self.file_path.split('/')[-1].split('.')[0].lower() + '_' + self.model_type # bgd_Education_1001_5000_LSTM
        self.new_dir = f'{self.dir}/{self.model_name}/'
        self.model_path = f'{self.new_dir}/{self.model_name}.pth'
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        if not os.path.exists(self.new_dir): os.makedirs(self.new_dir)
        self.next(self.ingest)

    @card
    @step
    def ingest(self):
        logger.info('Fetching dataframe..')
        if self.file_path[-3:] == 'csv':
            self.df = pd.read_csv(self.file_path, index_col='timestamp', parse_dates=True)
        elif self.file_path[-7:] == 'parquet':
            self.df = pd.read_parquet(self.file_path, index_col='timestamp', parse_dates=True)
        else:
            raise ValueError('Invalid format, only csv or parquet files are supported.')
        logger.info('Dataframe fetching complete..')
        self.next(self.pick_2017_data)

    @step
    def pick_2017_data(self):
        logger.info('Picking 2017 data..')
        self.df = self.df[self.df.index.year == 2017]
        self.next(self.fill_missing_data)

    @step
    def fill_missing_data(self):
        logger.info('Filling missing data points using weighted average..')
        # replace nan values with 0 in 2017 data
        self.df = self.df.replace(0, np.nan)
        # check number of missing values in 2017 data
        self.missing_values = find_nan_counts(self.df)
        # if missing values in 2017 data is high, fill missing data with weighted average
        if (self.missing_values>0): self.df = fill_missing_data(self.df)
        # todo: consider 2016 data as well in v2, can use a var to track which type of data is used.
        self.next(self.find_median)

    @step
    def find_median(self):
        logger.info('Computing and using median..')
        self.df_median = pd.DataFrame(self.df.apply(lambda x: x.median(), axis=1).to_frame(name='electricity'))
        self.next(self.aggregate_data)

    @card
    @step
    def aggregate_data(self):
        if self.use_hourly:
            logger.info('Skipping.. data is on hourly basis.')
            pass
        else:
            logger.info('Aggregating data on daily basis..')
            self.data = self.df_median.resample('D').sum() #daily data
        self.next(self.split_data)

    @card
    @step
    def split_data(self):
        logger.info('Splitting data into test and train...')
        self.train_data, self.test_data = train_test_split(
            self.df_median, test_size=0.2, shuffle=False
        )
        self.next(self.make_x_y)

    @step
    def make_x_y(self):
        self.X_train, self.y_train = sliding_windows(data=pd.DataFrame(self.train_data), lookback=self.lookback)
        self.X_test, self.y_test = sliding_windows(data=pd.DataFrame(self.test_data), lookback=self.lookback)
        logger.info(f'Shape of X_train: {self.X_train.shape}\nShape of y_train: {self.y_train.shape}\nShape of X_test: {self.X_test.shape}\nShape of y_test: {self.y_test.shape}')
        self.next(self.cross_validate_train)

    @card
    @step
    def cross_validate_train(self):
        logger.info(f'Creating {self.model_name}')
        self.model = LSTMModel(input_size=1)
        self.model.name = self.model_name
        self.tscv = TimeSeriesSplit(n_splits=5)
        self.train_losses, self.val_losses, self.mape_scores = [], [], []

        logger.info('Performing cross validation..')
        for i, (train_idx, val_idx) in enumerate (self.tscv.split(self.X_train)):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            mm_X = MinMaxScaler()
            mm_y = MinMaxScaler()

            X_train_normal = mm_X.fit_transform(X_train_fold.reshape(-1, 1)).reshape(-1, self.lookback, 1)
            X_val_normal = mm_X.transform(X_val_fold.reshape(-1, 1)).reshape(-1, self.lookback, 1)
            y_train_normal = mm_y.fit_transform(y_train_fold.reshape(-1, 1))
            y_val_normal = mm_y.transform(y_val_fold.reshape(-1, 1))

            train_dataset = ElecDataset(X_train_normal, y_train_normal)
            val_dataset = ElecDataset(X_val_normal, y_val_normal)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            # criterion = nn.MSELoss()
            criterion = nn.L1Loss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            num_epochs = self.epochs
            for epoch in range(num_epochs):
                self.model.train()
                epoch_train_loss = 0
                for sequences, targets in train_loader:
                    optimizer.zero_grad()
                    output = self.model(sequences)
                    loss = criterion(output, targets.view(-1, 1))
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)

                # Validation loss
                self.model.eval()
                epoch_val_loss = 0
                val_true = []
                val_preds = []
                with torch.no_grad():
                    for sequences, targets in val_loader:
                        output = self.model(sequences)
                        loss = criterion(output, targets.view(-1, 1))
                        epoch_val_loss += loss.item()
                        val_preds.extend(output.view(-1).tolist())
                        val_true.extend(targets.view(-1).tolist())

                avg_val_loss = epoch_val_loss / len(val_loader)
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

            val_preds = np.array(val_preds).reshape(-1, 1)
            val_true = np.array(val_true).reshape(-1, 1)
            val_preds_original = mm_y.inverse_transform(val_preds)
            val_true_original = mm_y.inverse_transform(val_true)

            mape = np.mean(np.abs((val_true_original - val_preds_original) / val_true_original)) * 100
            self.mape_scores.append(mape)
            logger.info(f'Fold {i}, MAPE: {mape:.2f}%')

            plt.figure(figsize=(12, 6))
            plt.plot(range(len(val_preds_original)), val_preds_original, label='Predictions')
            plt.plot(range(len(val_true_original)), val_true_original, label='True Values')
            plt.title(f'Fold {i} - Validation Predictions vs True Values - {self.model_name}')
            plt.legend()
            plot_name = f'{self.new_dir}/train_f{i}_{self.model_name}.png'
            plt.savefig(plot_name)
            logger.info(f'Training image saved at {plot_name}')
        self.next(self.model_scores)

    @card
    @step
    def model_scores(self):
        avg_train_loss_across_folds = sum(self.train_losses) / len(self.train_losses)
        avg_val_loss_across_folds = sum(self.val_losses) / len(self.val_losses)
        avg_mape_across_folds = sum(self.mape_scores) / len(self.mape_scores)
        logger.info(f'Average Train Loss across all folds: {avg_train_loss_across_folds}')
        logger.info(f'Average Validation Loss across all folds: {avg_val_loss_across_folds}')
        logger.info(f'Average MAPE across all folds: {avg_mape_across_folds:.2f}%')
        self.next(self.validate_model)

    @step
    def validate_model(self):
        logger.info('Validating LSTM on test set..')
        scaler = MinMaxScaler()
        scaler.fit(self.train_data[['electricity']])

        X_test_normal = scaler.transform(self.X_test.reshape(-1, 1)).reshape(-1, self.lookback, 1)
        y_test_normal = scaler.transform(self.y_test.reshape(-1, 1))

        test_dataset = ElecDataset(X_test_normal, y_test_normal)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.model.eval()
        test_pred, test_true = [], []

        with torch.no_grad():
            for sequences, targets in test_loader:
                output = self.model(sequences)
                test_pred.extend(output.view(-1).tolist())
                test_true.extend(targets.view(-1).tolist())

        test_pred = np.array(test_pred).reshape(-1, 1)
        test_true = np.array(test_true).reshape(-1, 1)
        test_pred_inv = scaler.inverse_transform(test_pred)
        test_true_inv = scaler.inverse_transform(test_true)

        test_mape = np.mean(np.abs((test_true_inv - test_pred_inv) / test_true_inv)) * 100
        logger.info(f'Test Set MAPE: {test_mape:.2f}%')

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(test_pred_inv)), test_pred_inv, label='Predictions')
        plt.plot(range(len(test_true_inv)), test_true_inv, label='True Values')
        plt.title(f'Test Set Predictions vs True Values - {self.model_name}')
        plt.legend()
        plot_name = f'{self.new_dir}/test_{self.model_name}.png'
        plt.savefig(plot_name)
        logger.info(f'Test image saved at {plot_name}')
        self.next(self.log_model)

    @card
    @step
    def log_model(self):
        logger.info(f'Logging model {self.model.name}')
        torch.save(self.model, self.model_path)
        logger.info(f'Model logged at {self.model_path}')
        self.next(self.end)

    @step
    def end(self):
        # logger.info('Saving all class vars as a JSONs.')
        # variables_dict = {key: value for key, value in self.__dict__.items()}
        # jsonstr = json.dumps(variables_dict, indent=4)
        # jsonfilename = f'{self.new_dir}/json_{self.model_name}.json'
        # with open(jsonfilename, 'w') as jsonfile:
        #     jsonfile.write(jsonstr)
        logger.info('Pipeline end.')

    def check_existing_successful_run(self):
        flow = Flow('BGD_LinearFlow')
        for run in flow.runs():
            if run.successful and run.data.model_name == self.model_name:
                return run
        return None

# NON-Class functions:
#UNUSED for now:
# def check_if_year_has_data(df:pd.DataFrame, year:int=2016, threshold:int=0.2)->bool:
#     specified_year_data = df[df.index.year==year]
#     zero_counts = (specified_year_data==0).sum().sum()
#     total_count = zero_counts.size
#     percentage_counts = zero_counts/total_count
#     return percentage_counts > threshold
#     # if zero % is greater than threshold then skip this column

def find_nan_counts(df: pd.DataFrame)->int:
    # Count the total number of NaN values in the entire DataFrame
    nan_counts_per_column = df.isna().sum()
    total_nan_count = df.isna().sum().sum()

    logger.info("NaN counts per column:")
    logger.info(nan_counts_per_column)
    logger.info("\nTotal NaN count in the DataFrame:", total_nan_count)
    return total_nan_count

# Function to compute the weighted average for a given missing index
def fill_missing_data(df):

    def weighted_average(index, column, max_neighbors=10):
        neighbors = []
        weights = []

        # Search for up to `max_neighbors` data points forward and backward
        for i in range(1, max_neighbors + 1):
            if index - pd.DateOffset(hours=i) in df.index and not pd.isnull(df.at[index - pd.DateOffset(hours=i), column]):
                neighbors.append(df.at[index - pd.DateOffset(hours=i), column])
                weights.append(1 / i)
            if index + pd.DateOffset(hours=i) in df.index and not pd.isnull(df.at[index + pd.DateOffset(hours=i), column]):
                neighbors.append(df.at[index + pd.DateOffset(hours=i), column])
                weights.append(1 / i)

        if neighbors:
            weighted_sum = sum(neighbor * weight for neighbor, weight in zip(neighbors, weights))
            return weighted_sum / sum(weights)
        else:
            return np.nan

    # Iterate over each column
    for column in df.columns:
        # Find indices where data is missing
        missing_indices = df[column][df[column].isnull()].index

        # Fill missing data
        for index in missing_indices:
            df.at[index, column] = weighted_average(index, column)

    return df


def sliding_windows(data:pd.DataFrame, lookback:int)->tuple[np.array, np.array]:
  X, y = [], []
  for i in range(len(data) - lookback):
    X.append(data.iloc[i:i + lookback].values)
    y.append(data.iloc[i + lookback])
  return np.array(X), np.array(y)

if __name__ == '__main__':
    BGD_LinearFlow()
