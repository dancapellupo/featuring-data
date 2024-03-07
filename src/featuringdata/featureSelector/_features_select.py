
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.sklearn import XGBRegressor

from ._recursive_fit import recursive_fit


class FeatureSelector:

    def __init__(self, numeric_cols, non_numeric_cols, report_prefix='FeatureSelection', target_col=None,
                 test_size=0.15, parameter_dict=None):

        self.numeric_cols = numeric_cols
        self.non_numeric_cols = non_numeric_cols
        self.report_prefix = report_prefix
        self.target_col = target_col

        self.test_size = test_size

        if parameter_dict is None:
            self.parameter_dict = {'max_depth': [3, 4, 5, 6], 'gamma': [0, 1, 5],
                                   'min_child_weight': [0, 1, 5], 'max_delta_step': [0, 1, 5]}
        else:
            self.parameter_dict = parameter_dict

        self.hyperparams_df = pd.DataFrame()

    def run(self, data_df):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_folder = './{}_ModelTraining_plots_{}'.format(self.report_prefix, timestamp)
        # Path(plots_folder).mkdir()

        # all_columns = self.numeric_cols.extend(self.non_numeric_cols)

        X = data_df[self.numeric_cols]

        X_onehot = pd.get_dummies(data_df[self.non_numeric_cols], dtype=int)

        X = X.merge(X_onehot, left_index=True, right_index=True)

        y = data_df[self.target_col].values
        y = np.log1p(data_df[self.target_col].values)

        X_train_42, X_test_42, y_train_42, y_test_42 = train_test_split(X, y, test_size=self.test_size,
                                                                        random_state=42)
        X_train_46, X_test_46, y_train_46, y_test_46 = train_test_split(X, y, test_size=self.test_size,
                                                                        random_state=46)

        # TODO Allow user to set max/min values of these hyperparam ranges, as well as number of total iterations,
        #  which would define how many values to consider per hyperparam

        X_train_comb = [X_train_42, X_train_46]
        X_test_comb = [X_test_42, X_test_46]

        y_train_comb = [y_train_42, y_train_46]
        y_test_comb = [y_test_42, y_test_46]

        training_results_df, self.hyperparams_df = recursive_fit(X_train_comb, y_train_comb, X_test_comb, y_test_comb,
                                                                 parameter_dict=self.parameter_dict)

        best_result_ind_1 = np.argmin(training_results_df["RMSE_test_1"].values)
        best_result_ind_2 = np.argmin(training_results_df["RMSE_test_2"].values)

        best_result_1 = training_results_df["RMSE_test_1"].values[best_result_ind_1]
        best_result_2 = training_results_df["RMSE_test_2"].values[best_result_ind_2]

        print('Best results: (1) {} [{}], (2) {} [{}]\n'.format(
            best_result_1, best_result_ind_1, best_result_2, best_result_ind_2))

        if best_result_1 < best_result_2:
            data_ind = 0
            best_ind = best_result_ind_1
        else:
            data_ind = 1
            best_ind = best_result_ind_2

        X_train_best = X_train_comb[data_ind][training_results_df.loc[
            best_ind, "feature_list_{}".format(data_ind+1)].split(', ')]
        X_test_best = X_test_comb[data_ind][training_results_df.loc[
            best_ind, "feature_list_{}".format(data_ind+1)].split(', ')]

        hyperparam_iters = self.hyperparams_df.index.values
        hyperparam_iter = hyperparam_iters[np.where((best_ind - hyperparam_iters) >= 0)[0][-1]]

        hyperparams_dict = self.hyperparams_df.loc[hyperparam_iter].to_dict()
        print('Using Iter {} from data split {} with {}'.format(best_ind, data_ind, hyperparams_dict))

        # XGBoost Training:
        xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42, **hyperparams_dict)
        xgb_reg.fit(X_train_best, y_train_comb[data_ind], eval_set=[(X_test_best, y_test_comb[data_ind])], verbose=True)

        y_test_pred = xgb_reg.predict(X_test_best)
        print(mean_absolute_error(np.expm1(y_test_comb[data_ind]), np.expm1(y_test_pred)))

        training_results_df.to_csv('{}_training_results_full_{}.csv'.format(self.report_prefix, timestamp))
        self.hyperparams_df.to_csv('{}_best_hyperparameters_{}.csv'.format(self.report_prefix, timestamp))

        return training_results_df


