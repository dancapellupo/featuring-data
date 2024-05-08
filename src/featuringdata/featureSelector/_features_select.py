
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor, XGBClassifier

from ._create_pdf_report import (
    initialize_pdf_doc,
    add_text_pdf,
    add_plot_pdf,
    save_pdf_doc
)

from ._recursive_fit import recursive_fit, calc_model_metric, get_metric_names

from ._generate_plots import plot_inline_scatter, plot_xy, convert_title_to_filename, plot_horizontal_line, plot_vertical_line, save_fig, plot_xy_splitaxis


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
                 target_log=False, target_type='regression', test_size=0.15, parameter_dict=None):

        self.numeric_cols = numeric_cols
        self.non_numeric_cols = non_numeric_cols
        self.report_prefix = report_prefix
        self.target_col = target_col
        self.target_log = target_log
        self.target_type = target_type

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

    def run(self, data_df, master_columns_df=None, numeric_df=None, non_numeric_df=None):
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
        if self.target_type == 'classification':
            enc = LabelEncoder()
            self.y = enc.fit_transform(data_df[self.target_col].values)
        elif self.target_log:
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
            target_type=self.target_type, parameter_dict=self.parameter_dict)

        # --------------------------------------------------------------------
        # Identify Best Results

        primary_metric, secondary_metric = get_metric_names(target_type=self.target_type)

        # TODO: Identify best run based on metric out to certain number [3?] of decimal points
        best_result_ind_1 = np.argmin(training_results_df[f"{primary_metric}_test_1"].values)
        best_result_ind_2 = np.argmin(training_results_df[f"{primary_metric}_test_2"].values)

        best_result_1 = training_results_df[f"{primary_metric}_test_1"].values[best_result_ind_1]
        best_result_2 = training_results_df[f"{primary_metric}_test_2"].values[best_result_ind_2]

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

        if self.target_type == 'regression':
            xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42, **hyperparams_dict)
        else:
            xgb_reg = XGBClassifier(n_estimators=1000, early_stopping_rounds=20, random_state=42, **hyperparams_dict)
        xgb_reg.fit(X_train_best, y_train_comb[data_ind], eval_set=[(X_test_best, y_test_comb[data_ind])], verbose=True)

        y_test_pred = xgb_reg.predict(X_test_best)
        
        if self.target_log:
            sec_metric_final = calc_model_metric(np.expm1(y_test_comb[data_ind]), np.expm1(y_test_pred),
                                          target_type=self.target_type, metric_type='easy')
        else:
            sec_metric_final = calc_model_metric(y_test_comb[data_ind], y_test_pred, target_type=self.target_type,
                                          metric_type='easy')
        print(f'\nFinal {secondary_metric}: {sec_metric_final:.3f}\n')

        # --------------------
        # Save results to CSV:
        training_results_df.to_csv('{}_training_results_full_{}.csv'.format(self.report_prefix, timestamp))
        self.hyperparams_df.to_csv('{}_best_hyperparameters_{}.csv'.format(self.report_prefix, timestamp))

        # --------------------------------------------------------------------
        # Generating PDF Document and Plots:
        self.pdf = initialize_pdf_doc()

        # --------------------------------------------------------------------
        # PLOTS #1 and #2 - Plot of model metric vs iteration

        # First look for large gaps along x-axis:
        num_features_start = training_results_df["num_features_{}".format(data_ind+1)].iloc[0]
        gap_loc = np.where(
            np.diff(training_results_df["num_features_{}".format(data_ind+1)].values)[0:5] < -0.2*num_features_start)[0]
        start_ii = gap_loc[-1] + 1 if gap_loc.size > 0 else 0

        # ---
        # PLOT #1 - Primary metric used in xgboost training:

        # Create the plots:
        f, ax = plot_inline_scatter(training_results_df.iloc[start_ii:], x_col=f"num_features_{1}",
                                    y_col=f"{primary_metric}_test_{1}", leg_label=f'Data Split {1}', outfile=False)
        best_prim_metric = training_results_df[f"{primary_metric}_test_{data_ind+1}"].iloc[best_ind]
        if primary_metric == 'RMSE':
            ylabel = 'RMSE for Val Set'
        else:
            ylabel = 'Logloss for Val Set'
        plot_inline_scatter(training_results_df.iloc[start_ii:], f=f, ax=ax, x_col=f"num_features_{2}",
                            y_col=f"{primary_metric}_test_{2}", leg_label=f'Data Split {2}',
                            xlabel='Number of Features in Iteration', ylabel=ylabel, hline=best_prim_metric,
                            vline=training_results_df[f"num_features_{data_ind+1}"].iloc[best_ind],
                            reverse_x=True, overplot=True, outfile=True, plots_folder=plots_folder,
                            title=f'num_features_vs_{primary_metric}')

        # Add plot and informative text to PDF:
        self.pdf = add_text_pdf(self.pdf, txt="Recursive Training Results", style='B', space_below=10)
        self.pdf = add_plot_pdf(self.pdf, file_path=plots_folder+f'/num_features_vs_{primary_metric}'+'.png',
                                new_page=False)
        if start_ii > 0:
            out_txt = (f"Note: The point from the first iteration with {num_features_start} features and a"
                       f"{primary_metric} of {training_results_df[f'{primary_metric}_test_{data_ind+1}'].iloc[0]} was "
                       f"removed fromthis plot.")
            self.pdf = add_text_pdf(self.pdf, txt=out_txt)
            out_txt = ("Normally, the way this recursive model training works is that it removes the feature with the "
                       "lowest importance at each iteration. However, if there are multiple features that have exactly "
                       "zero importance, then all of those zero importance features are removed at once.")
            self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        out_txt = ("The above plot has our model metric on the y-axis, and the number of features for each model "
                   "training iteration on the x-axis. In other words, each dot here represents an iteration of the "
                   "recursive model training.")
        self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        out_txt = "As the number of features is reduced, eventually the model will start to perform much more poorly."
        self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        # TODO: Put metric values in boldface:
        out_txt = (f"This plot shows the primary metric that was used in model training, which is {primary_metric}. "
                   f"The vertical line is the location with the best value of this metric, which is a {primary_metric} "
                   f"of ")
        self.pdf = add_text_pdf(self.pdf, txt=out_txt, space_below=0)
        self.pdf = add_text_pdf(self.pdf, style='B', txt=f"{best_prim_metric}", space_below=0)
        self.pdf = add_text_pdf(self.pdf, txt=f", compared to the starting {primary_metric} of ", space_below=0)
        self.pdf = add_text_pdf(self.pdf, style='B', space_below=0,
                                txt=f"{training_results_df[f'{primary_metric}_test_{data_ind+1}'].iloc[0]}")
        self.pdf = add_text_pdf(self.pdf, txt=f".")
        # DONE: Make italic:
        out_txt = f"Note that lower values of {primary_metric} indicate better model performance."
        self.pdf = add_text_pdf(self.pdf, style='I', txt=out_txt)
        out_txt = (f"The model training started with {num_features_start} features (after one-hot encoding any "
                   f"categorical features), and achieved the best model training results with ")
        self.pdf = add_text_pdf(self.pdf, txt=out_txt, space_below=0)
        self.pdf = add_text_pdf(
            self.pdf, txt=f"{training_results_df['num_features_{}'.format(data_ind+1)].iloc[best_ind]} features",
            style='B', space_below=0)
        self.pdf = add_text_pdf(self.pdf, txt=f".")
        # TODO: Add baseline for logloss - random assignment based on fraction with each label
        # TODO: Assess how low, in terms of number of features, one could go without drastically decreasing the
        #  performance
        # TODO: Print MAE compared to range of y_val valuee
        
        # ---
        # PLOT #2 - Secondary evaluation metric:

        if secondary_metric == 'MAE':
            best_sec_metric_ind = np.argmin(training_results_df[f"{secondary_metric}_test_{data_ind+1}"].values)
        else:
            best_sec_metric_ind = np.argmax(training_results_df[f"{secondary_metric}_test_{data_ind+1}"].values)
        best_sec_metric = training_results_df[f"{secondary_metric}_test_{data_ind+1}"].iloc[best_sec_metric_ind]

        f, ax = plot_inline_scatter(training_results_df.iloc[start_ii:], x_col=f"num_features_{1}",
                                    y_col=f"{secondary_metric}_test_{1}", leg_label=f'Data Split {1}', outfile=False)
        if secondary_metric == 'MAE':
            ylabel = 'Mean Average Error (MAE) for Val Set'
        else:
            ylabel = 'Cohen-Kappa for Val Set'
        plot_inline_scatter(training_results_df.iloc[start_ii:], f=f, ax=ax, x_col=f"num_features_{2}",
                            y_col=f"{secondary_metric}_test_{2}", leg_label=f'Data Split {2}',
                            xlabel='Number of Features in Iteration', ylabel=ylabel, hline=best_sec_metric,
                            vline=training_results_df[f"num_features_{data_ind+1}"].iloc[best_sec_metric_ind],
                            reverse_x=True, overplot=True, outfile=True, plots_folder=plots_folder,
                            title=f'num_features_vs_{secondary_metric}')
        
        self.pdf = add_plot_pdf(self.pdf, file_path=plots_folder+f'/num_features_vs_{secondary_metric}'+'.png',
                                new_page=True)
        out_txt = f"This plot is the same as the previous page, except with the secondary metric, {secondary_metric}."
        self.pdf = add_text_pdf(self.pdf, txt=out_txt)
        out_txt = "It is therefore possible that the 'best' iteration will be different in this plot."
        self.pdf = add_text_pdf(self.pdf, style='I', txt=out_txt)
        self.pdf = add_text_pdf(self.pdf, txt=f"The best value of {secondary_metric} is ", space_below=0)
        self.pdf = add_text_pdf(self.pdf, style='B', txt=f"{best_sec_metric}", space_below=0)
        self.pdf = add_text_pdf(self.pdf, txt=f", compared to the starting value of ", space_below=0)
        self.pdf = add_text_pdf(self.pdf, style='B', space_below=0,
                                txt=f"{training_results_df[f'{secondary_metric}_test_{data_ind+1}'].iloc[0]}")
        self.pdf = add_text_pdf(self.pdf, txt=f".")
        if secondary_metric == 'CohKap':
            out_txt = (f"Note that with the Cohen-Kappa score, the possible range is 0 to 1, with 0 meaning that the "
                       f"model has no predictive power and 1 is the best it could be.")
            self.pdf = add_text_pdf(self.pdf, style='I', txt=out_txt)
        self.pdf = add_text_pdf(
            self.pdf, txt="With this metric, the best model training result occurred with ", space_below=0)
        self.pdf = add_text_pdf(
            self.pdf, txt=f"{training_results_df['num_features_{}'.format(data_ind+1)].iloc[best_ind]} features",
            style='B', space_below=0)
        self.pdf = add_text_pdf(self.pdf, txt=f".")
        
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

        self.pdf = add_text_pdf(self.pdf, txt="Exploring Feature Importance during Iterative Training", style='B',
                                new_page=True, space_below=10)

        num_features = training_results_df["num_features_{}".format(data_ind+1)].values
        # Set the number of features to show on each plot:
        num_feat_per_plot = 5
        # Set the total number of features to plot:
        tot_feat_to_plot = min(20, len(self.feat_import_bycol_df) - len(self.feat_import_bycol_df) % num_feat_per_plot)
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

                print(jj, jjj, col)

                # If this is the first feature for this plot panel:
                if jjj == 0:
                    f, ax = plot_xy(x, y, xlabel='Number of Features (Data Split {})'.format(data_ind+1),
                                    ylabel='Feature Importance', overplot=False, outfile=False, markersize=5, label=col)
                elif jjj < num_feat_per_plot-1:
                    f, ax = plot_xy(x, y, f=f, ax=ax, overplot=True, outfile=False, markersize=5, label=col)
                # If this is the last feature for this plot panel, save the
                #  plot to disk:
                else:
                    plot_xy(x, y, f=f, ax=ax, overplot=True, outfile=True, plots_folder=plots_folder, reverse_x=True,
                            title='feature_importance_vs_number_features_{}'.format(jj), markersize=5, label=col)

            # Create a new PDF page for every 2 plots:
            new_page = True if (jj != 0) and ((jj % 2) == 0) else False
            self.pdf = add_plot_pdf(
                self.pdf, file_path=plots_folder+'/feature_importance_vs_number_features_{}'.format(jj)+'.png',
                new_page=new_page)

        # --------------------------------------------------------------------
        # PLOT #3 - Generate plot of feature importance versus correlation
        #  with target variable

        cols_best_iter = self.feat_import_bycol_df.dropna().index
        # This plot can only be generated if a dataframe with feature
        #  correlations is passed to the function:
        if master_columns_df is not None:
            self.pdf = add_text_pdf(
                self.pdf, txt="Exploring Feature Importance Compared to Individual Feature Correlations", style='B',
                new_page=True, space_below=10)
            plot_title = 'Target Correlation vs Feature Importance'

            # Get a list of features that are both numeric and are part of the
            #  "best" training iteration:
            numeric_df = master_columns_df.loc[master_columns_df["Column Type"] == 'numeric']
            if len(numeric_df) > 0:
                numeric_best_feat = set(cols_best_iter).intersection(set(numeric_df.index))
                print('Number of numeric features in best iteration: {}'.format(len(numeric_best_feat)))

                x, y = [], []
                for feat in numeric_best_feat:
                    x.append(numeric_df.loc[feat, "Random Forest"])
                    y.append(self.feat_import_bycol_df.loc[feat, "best_feat_imp"])

                f, ax = plot_xy(x, y, xlabel='RF Correlation between Feature and Target', ylabel='Feature Importance',
                                overplot=False, outfile=False, plots_folder=plots_folder, title=plot_title,
                                markersize=5, label='Numeric Feature')

            non_numeric_df = master_columns_df.loc[master_columns_df["Column Type"] == 'non-numeric']
            if len(non_numeric_df) > 0:
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
                plot_xy(x, y, f=f, ax=ax, overplot=True, outfile=True, plots_folder=plots_folder, title=plot_title,
                        markersize=5, label='Non-Numeric Feature')
            
            self.pdf = add_plot_pdf(self.pdf, file_path=plots_folder+'/'+convert_title_to_filename(plot_title)+'.png',
                                    new_page=False)

        # Save PDF document to current working directory
        save_pdf_doc(self.pdf, custom_filename=self.report_prefix, timestamp=timestamp)

        return training_results_df

    def rerun_plots(self):
        # TODO: Option to re-run xgboost and plots with a different choice of iteration from recursive run
        pass


