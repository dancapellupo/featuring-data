
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.sklearn import XGBRegressor

from ._create_pdf_report import (
    initialize_pdf_doc,
    add_text_pdf,
    add_plot_pdf,
    save_pdf_doc
)

from ._recursive_fit import recursive_fit

from ._generate_plots import plot_inline_scatter, plot_xy, plot_horizontal_line, plot_vertical_line, save_fig, plot_xy_splitaxis


class FeatureSelector:
    """
    This class implements an iterative machine learning model training
    (currently using the xgboost algorithm) to help with feature selection and
    understanding the importance of the input features.

    The results of this iterative training are available within your Jupyter
    Notebook environment for further analysis and model training tasks. This
    code should be used with your training set, with your holdout/test set
    kept separate. This code will separate your training set into several
    training / validation set splits [currently set at two separate splits].

    Just as in the EDA class of this library, a (separate) nicely formatted
    PDF report is generated and saved in your current working directory - for
    easy reference or sharing results with team members and even stakeholders.
    The PDF report includes also explanations of the generated plots.

    The functions within this class perform the following tasks:
    - Data preparation tasks:
        - Perform one-hot encoding on categorical features.
        - Split the data into [at least] two training and validation splits.
    - Iterative / recursive model training:
        - There are a number of feature selection techniques (see the Readme
          for more details), but after some testing, this author recommends
          the recursive technique where one starts training with all features,
          and then removes the feature with the lowest feature importance at
          each iteration. The relevant model metric (e.g., mean squared error
          for regression) is saved at each iteration, and at the end we can
          see how many, and which, features give as good, if not, better
          results than using all features.
        - Another important part of model training is selecting
          hyperparameters. This code utilizes a grid search approach, and
          performs this search a few times during the iterative training, to
          take into account the possibility that a better set of
          hyperparameters may exist when training with a smaller number of
          features than with all features.
        - This iterative training is performed on at least two different
          random splits of the data.
        - The following information is kept from every iteration: the feature
          importance values of every feature at every iteration, performance
          metrics on both the training and validation set, the number of
          features, and the features removed at the end of each iteration.

    Parameters
    ----------
    numeric_cols : List
        The final list of numeric column names after any adjustments are made
        based on the number of unique values (from featuringdata.featuresEDA).

    non_numeric_cols : List
        The final list of non-numeric / categorical columns names after any
        adjustments / additions from columns originally treated as numerical
        (from featuringdata.featuresEDA).

    report_prefix : str, default='FeatureSelection'
        The prefix used for filename of report, as well as the name of the
        folder containing the PNG files of plots.

    target_col : str, default=None
        The name of the dataframe column containing the target variable.

    target_log : bool, default=False
        Whether to take the log of the target variable for model training.

    TODO: Change name to 'val_size'
    test_size : float, default=0.15
        Fraction of the input data to use for validation set.

    parameter_dict : dict, default=None
        Dictionary of hyperparameters with a range of parameter values to use
        for the hyperparameter grid search.

    Attributes
    ----------
    pdf : ??
        The PDF object.

    X : pd.DataFrame
        The final dataframe used for model training, after the data
        preparation steps, and with all features, including the one-hot
        encoding.

    y : pd.Series
        The target variable.

    hyperparams_df : pd.DataFrame
        The best hyperparameters at each iteration where hyperparameter search
        is performed.

    feature_importance_dict_list : list
        A list of dictionaries, with each list corresponding to a different
        split of the data. The dictionaries contain a list of feature
        importance values for each feature, corresponding to the iterations in
        which each feature appears. In other words, if a feature appeared in
        only the first iteration, then the list for that features contains
        just 1 feature importance values. If the feature appeared in the first
        50 iterations before being removed, then its list would contain 50
        feature importance values.

    feat_import_bycol_df : pd.DataFrame
        This dataframe summarizes the feature importance information for each
        feature, with the following columns:
        - "max_feat_imp": The maximum feature importance that each feature had
            during the iterative training process.
        - "best_feat_imp": The feature importance value of each feature at the
            iteration with the best model performance (if a feature was
            removed before this iteration, then the value here is set to
            np.nan.
        - "num_iters": The number of iterations each feature appeared in,
            before being removed.


    """

    def __init__(self, numeric_cols, non_numeric_cols, report_prefix='FeatureSelection', target_col=None,
                 target_log=False, test_size=0.15, parameter_dict=None):

        self.numeric_cols = numeric_cols
        self.non_numeric_cols = non_numeric_cols
        self.report_prefix = report_prefix
        self.target_col = target_col
        self.target_log = target_log

        self.test_size = test_size

        if parameter_dict is None:
            self.parameter_dict = {'max_depth': [3, 4, 5, 6], 'gamma': [0, 1, 5],
                                   'min_child_weight': [0, 1, 5], 'max_delta_step': [0, 1, 5]}
        else:
            self.parameter_dict = parameter_dict

        self.pdf = None

        self.X = None
        self.y = None

        self.hyperparams_df = pd.DataFrame()
        self.feature_importance_dict_list = list()
        self.feat_import_bycol_df = pd.DataFrame()

    def run(self, data_df, numeric_df=None, non_numeric_df=None):
        """
        Run an iterative model training on a given dataset:

        This function runs the following steps:
        - Data preparation tasks
        - Iterative / recursive model training
        - Generate plots and a PDF report

        Parameters
        ----------
        data_df : pd.DataFrame of shape (n_samples, n data columns)
            The data to be analyzed.

        numeric_df : pd.DataFrame, optional (default=None)
            A dataframe with all numeric features and measures of their
            correlation with the target variable (from
            featuringdata.featuresEDA). This is used for comparing these
            correlations with model feature importance values.

        numeric_df : pd.DataFrame, optional (default=None)
            A dataframe with all non-numeric/categorical features and measures
            of their correlation with the target variable (from
            featuringdata.featuresEDA). This is used for comparing these
            correlations with model feature importance values.

        Returns
        -------
        training_results_df : pd.DataFrame
            A dataframe with comprehensive results of the iterative model
            training run.
            The index of the dataframe is the number of the iteration,
            starting from iteration 0 with all features included. The
            following columns are generated for each random data split:
            - "RMSE_train_":
            - "RMSE_test_":
            - "MAE_test_":
            - "num_features_":
            - "feature_list_":
            - "feat_high_import_name_":
            - "feat_high_import_val_":
            - "features_to_remove_":


        """

        # Save the current timestamp for the report filename and the folder to
        # contain the PNG plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # The directory is created for the EDA plots:
        plots_folder = './{}_ModelTraining_plots_{}'.format(self.report_prefix, timestamp)
        Path(plots_folder).mkdir()

        # all_columns = self.numeric_cols.extend(self.non_numeric_cols)

        # --------------------------------------------------------------------
        # Data Preparation

        # Numeric columns from the input dataset:
        self.X = data_df[self.numeric_cols]

        # Perform onehot encoding on any categorical features:
        if len(self.non_numeric_cols) > 0:
            X_onehot = pd.get_dummies(data_df[self.non_numeric_cols], dtype=int)

            self.X = self.X.merge(X_onehot, left_index=True, right_index=True)

        # Take the log of the target variable, if user chooses:
        if self.target_log:
            self.y = np.log1p(data_df[self.target_col].values)
        else:
            self.y = data_df[self.target_col].values

        # TODO: Update this code for more than 2 splits
        # Perform random data splits:
        X_train_42, X_test_42, y_train_42, y_test_42 = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                        random_state=42)
        X_train_46, X_test_46, y_train_46, y_test_46 = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                        random_state=46)

        # TODO Allow user to set max/min values of the hyperparam ranges, as well as number of total iterations,
        #  which would define how many values to consider per hyperparam

        X_train_comb = [X_train_42, X_train_46]
        X_test_comb = [X_test_42, X_test_46]

        y_train_comb = [y_train_42, y_train_46]
        y_test_comb = [y_test_42, y_test_46]

        # --------------------------------------------------------------------
        # Run Recursive Training

        training_results_df, self.hyperparams_df, self.feature_importance_dict_list = recursive_fit(
            X_train_comb, y_train_comb, X_test_comb, y_test_comb, target_log=self.target_log,
            parameter_dict=self.parameter_dict)

        # --------------------------------------------------------------------
        # Identify Best Results

        # TODO: Identify best run based on metric out to certain number [3?] of decimal points
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

        # Find the best hyperparameters relevant to the "best" iteration:
        hyperparam_iters = self.hyperparams_df.index.values
        hyperparam_iter = hyperparam_iters[np.where((best_ind - hyperparam_iters) >= 0)[0][-1]]

        hyperparams_dict = self.hyperparams_df.loc[hyperparam_iter].to_dict()
        print('Using Iter {} from data split {} with {}'.format(best_ind, data_ind+1, hyperparams_dict))

        # --------------------------------------------------------------------
        # XGBoost Training with "Best" Feature Selection

        xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42, **hyperparams_dict)
        xgb_reg.fit(X_train_best, y_train_comb[data_ind], eval_set=[(X_test_best, y_test_comb[data_ind])], verbose=True)

        y_test_pred = xgb_reg.predict(X_test_best)

        if self.target_log:
            mae_final = mean_absolute_error(np.expm1(y_test_comb[data_ind]), np.expm1(y_test_pred))
        else:
            mae_final = mean_absolute_error(y_test_comb[data_ind], y_test_pred)
        print('\nFinal MAE: {}\n'.format(mae_final))

        # --------------------
        # Save results to CSV:
        training_results_df.to_csv('{}_training_results_full_{}.csv'.format(self.report_prefix, timestamp))
        self.hyperparams_df.to_csv('{}_best_hyperparameters_{}.csv'.format(self.report_prefix, timestamp))

        # --------------------------------------------------------------------
        # Generating PDF Document and Plots:
        self.pdf = initialize_pdf_doc()

        # --------------------------------------------------------------------
        # PLOT #1 - Plot of model metric vs iteration

        # First look for large gaps along x-axis:
        num_features_start = training_results_df["num_features_{}".format(data_ind+1)].iloc[0]
        gap_loc = np.where(
            np.diff(training_results_df["num_features_{}".format(data_ind+1)].values)[0:5] < -0.2*num_features_start)[0]
        start_ii = gap_loc[-1] + 1 if gap_loc.size > 0 else 0

        # Creat the plot:
        f, ax = plot_inline_scatter(training_results_df.iloc[start_ii:], x_col="num_features_{}".format(1),
                                    y_col="MAE_test_{}".format(1), outfile=False)
        best_mae = training_results_df["MAE_test_{}".format(data_ind+1)].iloc[best_ind]
        plot_inline_scatter(training_results_df.iloc[start_ii:], f=f, ax=ax, x_col="num_features_{}".format(2),
                                    y_col="MAE_test_{}".format(2), xlabel='Number of Features in Iteration',
                                    ylabel='Mean Average Error (MAE) for Val Set',
                                    hline=best_mae,
                                    vline=training_results_df["num_features_{}".format(data_ind+1)].iloc[best_ind],
                                    overplot=True, outfile=True, plots_folder=plots_folder, title='num_features_vs_MAE')
        # ax = plot_horizontal_line(ax, y_loc=best_mae)
        # ax = plot_vertical_line(ax, x_loc=training_results_df["num_features_{}".format(data_ind+1)].iloc[best_ind])
        # save_fig(f, ax, plots_folder=plots_folder, title='num_features_vs_MAE')

        # plot_xy_splitaxis(x=training_results_df["num_features_1"].values, y=training_results_df["MAE_test_1"].values,
        #                   plots_folder=plots_folder, title='num_features_vs_MAE')

        # Add plot and informative text to PDF:
        self.pdf = add_text_pdf(self.pdf, txt="Recursive Training Results", bold=True)
        self.pdf = add_plot_pdf(self.pdf, file_path=plots_folder+'/num_features_vs_MAE'+'.png', new_page=False)
        if start_ii > 0:
            out_txt = ("Note: The point from the first iteration with {} features and an MAE of {} was removed from "
                       "this plot.").format(num_features_start,
                                            training_results_df["MAE_test_{}".format(data_ind+1)].iloc[0])
            self.pdf = add_text_pdf(self.pdf, txt=out_txt)
            out_txt = ("Normally, the way this recursive model training works is that it removes the feature with the "
                       "lowest importance at each iteration. However, if there are multiple features that have exactly "
                       "zero importance, then all of those zero importance features are removed at once.")
            self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        out_txt = ("The above plot has our model metric on the y-axis, and the number of features for each model "
                   "training iteration on the x-axis. In other words, each dot here represents an iteration of the "
                   "recursive model training.")
        self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        out_txt = ("As the number of features is reduced, eventually the model will start to perform much more poorly. "
                   "The vertical line is the location with the best value of the evaluation metric, which is an MAE of "
                   "{}), compared to the starting MAE of {}.").format(
            best_mae, training_results_df["MAE_test_{}".format(data_ind+1)].iloc[0])
        self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        out_txt = ("The model training started with {} features (after one-hot encoding any categorical "
                   "features), and achieved the best model training results with {} features.").format(
            num_features_start, training_results_df["num_features_{}".format(data_ind+1)].iloc[best_ind])
        self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        # TODO: Assess how low, in terms of number of features, one could go without drastically decreasing the
        #  performance
        # TODO: Print MAE compared to range of y_val values

        # ----------------------------------------------
        # Collect and examine feature importance values:
        self.feat_import_bycol_df = pd.DataFrame(columns=["max_feat_imp", "best_feat_imp", "num_iters"])
        for col in self.feature_importance_dict_list[data_ind].keys():
            feat_import_vals = self.feature_importance_dict_list[data_ind][col]
            best_feat_imp = feat_import_vals[best_ind] if best_ind < len(feat_import_vals) else np.nan
            self.feat_import_bycol_df.loc[col] = max(feat_import_vals), best_feat_imp, len(feat_import_vals)

        self.feat_import_bycol_df["num_iters"] = self.feat_import_bycol_df["num_iters"].astype(int)
        self.feat_import_bycol_df = self.feat_import_bycol_df.sort_values(by=["max_feat_imp"], ascending=False)

        # --------------------------------------------------------------------
        # PLOT #2 - Generate plots showing how the feature importance of the
        #  top features changes depending on the number of total features used

        num_features = training_results_df["num_features_{}".format(data_ind+1)].values
        # Set the number of features to show on each plot:
        num_feat_per_plot = 5
        # Set the total number of features to plot:
        tot_feat_to_plot = 20
        # Each loop corresponds to one plot:
        for jj in range(0, tot_feat_to_plot, num_feat_per_plot):
            cols_to_plot = self.feat_import_bycol_df.index[jj:jj+num_feat_per_plot]

            # Within each plot, loop over each feature to plot:
            for jjj, col in enumerate(cols_to_plot):
                # Find all the feature importance values for this feature
                #  (this will vary per feature depending on when that feature
                #   was removed during the iterative training):
                num_iters = int(self.feat_import_bycol_df.loc[col, "num_iters"])
                x = num_features[start_ii:num_iters]
                y = self.feature_importance_dict_list[data_ind][col][start_ii:]

                # If this is the first feature for this plot panel:
                if jjj == 0:
                    f, ax = plot_xy(x, y, xlabel='num_features_{}'.format(data_ind+1), ylabel='feature importance',
                                    leg_label=col, overplot=False, outfile=False)
                elif jjj < num_feat_per_plot-1:
                    f, ax = plot_xy(x, y, f=f, ax=ax, leg_label=col, overplot=True, outfile=False)
                # If this is the last feature for this plot panel, save the
                #  plot to disk:
                else:
                    plot_xy(x, y, f=f, ax=ax, leg_label=col, overplot=True, outfile=True, plots_folder=plots_folder,
                            title='feature_importance_vs_number_features_{}'.format(jj))

            # Create a new PDF page for every 2 plots:
            new_page = True if (jj % 2) == 0 else False
            self.pdf = add_plot_pdf(
                self.pdf, file_path=plots_folder+'/feature_importance_vs_number_features_{}'.format(jj)+'.png',
                new_page=new_page)

        # --------------------------------------------------------------------
        # PLOT #3 - Generate plot of feature importance versus correlation
        #  with target variable

        cols_best_iter = self.feat_import_bycol_df.dropna().index
        # This plot can only be generated if a dataframe with feature
        #  correlations is passed to the function:
        if numeric_df is not None:
            # Get a list of features that are both numeric and are part of the
            #  "best" training iteration:
            numeric_best_feat = set(cols_best_iter).intersection(set(numeric_df.index))
            print('Number of numeric features in best iteration: {}'.format(len(numeric_best_feat)))

            x, y = [], []
            for feat in numeric_best_feat:
                x.append(numeric_df.loc[feat, "Random Forest"])
                y.append(self.feat_import_bycol_df.loc[feat, "best_feat_imp"])

            f, ax = plot_xy(x, y, xlabel='RF Correlation between Feature and Target', ylabel='Feature Importance',
                            leg_label='Numeric Feature', overplot=False, outfile=False, plots_folder=plots_folder,
                            title='target_correlation_vs_feature_importance')

            if non_numeric_df is not None:
                feat_import_bycol_df_best = self.feat_import_bycol_df.dropna()
                # non_numeric_best_feat = set(cols_best_iter).difference()

                x, y = [], []
                for feat in non_numeric_df.index:
                    # For categorical features, the name may appear more than
                    #  once due to one-hot encoding:
                    if feat in cols_best_iter:
                        x.append(non_numeric_df.loc[feat, "RF_norm"])
                        y.append(self.feat_import_bycol_df.loc[feat, "best_feat_imp"])
                    else:
                        # For features with one-hot encoding, add up the
                        #  feature importance values for that feature:
                        feat_df = feat_import_bycol_df_best.loc[
                            feat_import_bycol_df_best.index.str.startswith(feat + '_')]
                        if len(feat_df) > 0:
                            x.append(non_numeric_df.loc[feat, "RF_norm"])
                            y.append(feat_df["best_feat_imp"].sum())

                print('Number of non-numeric features in best iteration: {}'.format(len(y)))
                plot_xy(x, y, f=f, ax=ax, leg_label='Non-Numeric Feature', overplot=True, outfile=True,
                        plots_folder=plots_folder, title='target_correlation_vs_feature_importance')

            self.pdf = add_plot_pdf(self.pdf, file_path=plots_folder+'/target_correlation_vs_feature_importance'+'.png',
                                    new_page=True)

        # Save PDF document to current working directory
        save_pdf_doc(self.pdf, custom_filename=self.report_prefix, timestamp=timestamp)

        return training_results_df

    def rerun_plots(self):
        # TODO: Option to re-run xgboost and plots with a different choice of iteration from recursive run
        pass


