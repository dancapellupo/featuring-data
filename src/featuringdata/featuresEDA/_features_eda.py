
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd

from ._create_pdf_report import (
    initialize_pdf_doc,
    section_on_null_columns,
    section_on_unique_values,
    section_on_target_column,
    section_on_target_column_plot,
    section_on_feature_corr,
    section_of_plots,
    save_pdf_doc
)

from ._initial_eda_functions import (
    count_null_values,
    sort_numeric_nonnumeric_columns,
    calc_column_summary_stats
)

from ._correlation import (
    calc_numeric_features_target_corr,
    calc_corr_numeric_features,
    calc_nonnumeric_features_target_corr
)

from ._generate_plots import (
    plot_hist,
    plot_hist_target_col,
    plot_feature_values
)


class FeaturesEDA:
    """
    This class implements Exploratory Data Analysis (EDA) on an input dataset.

    The results of the EDA are available within your Jupyter Notebook
    environment for further EDA and analysis, and a nicely formatted PDF
    report is generated and saved in your current working directory - for easy
    reference or sharing results with team members and even stakeholders.

    The functions within this class perform the following tasks:

     - Identifying data columns with NULL values and highlighting columns with
       the most NULL values.
        - Too many NULL values could indicate a feature that may not be worth
          keeping, or one may consider using a technique to fill NULL values.
        - It's worth noting that while many ML algorithms will not handle
          columns with NULL values, possibly throwing an error in the model
          training, xgboost, for example, does support NULL values (but it
          could still be worth filling those NULL values anyway).
     - A breakdown of numeric versus non-numeric/categorical features.
        - Any feature with only a single unique value is automatically removed
          from the analysis.
        - A feature that is of a numerical type (e.g, integers of 0 and 1),
          but have only two unique values are automatically considered as a
          categorical feature.
     - A count of unique values per feature.
        - Very few unique values in a column with a numerical type might
          indicate a feature that is actually categorical.
        - Too many unique values in a column with a non-numerical type (i.e.,
          an object or string) could indicate a column that maybe includes
          unique IDs or other information that might not be useful for an ML
          model. The PDF report will highlight these cases, to be noted for
          further review.
        - Furthermore, if a categorical feature has too many unique values, if
          one is considering using one-hot encoding, one should be aware that
          the number of actual features may increase by a lot when preparing
          your data for an ML model.
     - Feature Correlations
        - For both numeric and categorical features, the code will calculate
          the correlation between each feature and the target variable.
        - For numeric features, with a numeric target (i.e., a regression
          problem), the Pearson correlation is calculated.
        - For all features, a random forest model is run for each feature,
          with just that feature and the target variable. And the R^2 is
          reported as a proxy for correlation.
        - Optional: For numeric features, correlations between features are
          calculated. This can be very time-consuming for large numbers of
          features.
     - EDA Plots
        - For every feature, a plot of that feature versus the target variable
          is generated.
        - The code automatically selects the type of plot based on the number
          of unique values of that feature. For up to 10 unique values in a
          numeric feature, and for all categorical features, a box plot with a
          swarm plot is generated. If there are more than 1,000 data points,
          then only a random selection of 1,000 points are plotted on the
          swarm plot (but the box plot is calculated based on all points).
        - For typical numeric features, a standard scatter plot is generated.

    Parameters
    ----------
    report_prefix : str, default='FeatureSelection'
        The prefix used for filename of report, as well as the name of the
        folder containing the PNG files of plots.

    target_col : str, default=None
        The name of the dataframe column containing the target variable.

    cols_to_drop : list, default=None
        A list of column name(s) to drop from the dataframe before performing
        the EDA.

    numeric_uniq_vals_thresh : int, default=10
        For numeric features, any feature with fewer than this number of
        unique values will be reported.

    nonnumeric_uniq_vals_thresh : int, default=5
        For cateogorical features, any feature with greater than this number
        of unique values will be reported.

    Attributes
    ----------
    pdf : ??
        The PDF object.

    null_cols_df : pd.DataFrame
        A dataframe with all the features that have at least one NULL value.
        The dataframe has the following columns:
        - "Feature": Name of the data column.
        - "Num of Nulls": Total number of null values in the column.
        - "Frac Null": The fraction of all values in that column that are null.

    numeric_cols : List
        The final list of numeric column names after any adjustments are made
        based on the number of unique values.

    non_numeric_cols : List
        The final list of non-numeric / categorical columns names after any
        adjustments / additions from columns originally treated as numerical.

    numeric_uniq_vals_df : pd.DataFrame
        A dataframe listing any numeric columns that have no more than
        "numeric_uniq_vals_thresh" unique values. The dataframe has the
        following columns:
        - "Feature": Name of the numeric data column.
        - "Num Unique Values": The number of unique values in that data
          column.

    non_numeric_uniq_vals_df : pd.DataFrame
        A dataframe listing the number of unique values in every non-numeric
        column (here, we want to know if there is only a single unique value,
        and also if there are too many unique values). The dataframe has the
        following columns:
        - "Feature": Name of the non-numeric data column.
        - "Num Unique Values": The number of unique values in that data
          column.

    numeric_df : pd.DataFrame
        A dataframe with all numeric features (after any adjustments
        identifying originally numeric features as actually categorical) and 3
        different measures of their correlation with the target variable, for
        regression scenarios. For classification, ...
        The index of the dataframe is the feature/column names, and the
        columns are:
        - "Count not-Null": Number of non-null values for that feature.
        - "Pearson": The Pearson correlation between the feature and the
            target variable.
        - "Mutual Info" :
        - "Random Forest": The R^2 value when running a random forest model
            containing only this one feature and the target variable, using
            n_estimators=10.

    numeric_collinear_df = pd.DataFrame
        [Work in progress]

    numeric_collinear_summary_df = pd.DataFrame
        [Work in progress]

    non_numeric_df = pd.DataFrame
        A dataframe with all non-numeric/categorical features (after switching
        over any originally numeric features that are actually categorical)
        and a measure of their correlation with the target variable. The index
        of the dataframe is the feature/column names, and the columns are:
        - "Count not-Null": Number of non-null values for that feature.
        - "Num Unique": The number of unique values in that data column.
        - "Random Forest": The R^2 value when running a random forest model
            containing only this one feature and the target variable, using
            n_estimators=10. To be precise, each feature is split into
            multiple features for the random forest training using one-hot
            encoding.
        - "RF_norm": In a regression problem, the greater number of unique
            values that a categorical variable has will have a higher
            theoretical maximum R^2, so this R^2 is adjusted to more easily
            compare categorical features with different number of unique
            values (see documentation for further explanation).


    """

    def __init__(self, report_prefix='FeatureSelection', target_col=None, cols_to_drop=None,
                 numeric_uniq_vals_thresh=10, nonnumeric_uniq_vals_thresh=5):

        self.report_prefix = report_prefix
        self.target_col = target_col
        self.cols_to_drop = cols_to_drop
        self.numeric_uniq_vals_thresh = numeric_uniq_vals_thresh
        self.nonnumeric_uniq_vals_thresh = nonnumeric_uniq_vals_thresh

        self.pdf = None
        self.null_count_by_row_series = None
        self.master_columns_df = pd.DataFrame()
        self.numeric_cols = None
        self.non_numeric_cols = None
        self.numeric_collinear_df = pd.DataFrame()

    def run_initial_eda(self, data_df, output=True):
        """
        Run an initial exploratory data analysis (EDA) on a given dataset.

        This function runs the following steps:
        - Null values analysis
        - Unique values analysis
        - Switching numeric features to categorical

        It is worth running this function first on any new dataset, in order
        to identify any columns that should be removed before running the full
        correlation analysis (which can take a lot of time for many features
        and data samples).

        Parameters
        ----------
        data_df : Pandas dataframe of shape (n_samples, n data columns)
            The data to be analyzed.

        output : boolean, optional (default=True)
            Whether to save a PDF report of this initial EDA analysis.

        """

        # Save the current timestamp for the report filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # TODO: Add a check for columns with words like "ID" to suggest dropping them
        # Remove any columns that user identified as columns to ignore:
        if self.cols_to_drop is not None:
            data_df = data_df.drop(columns=self.cols_to_drop)

        # --------------------------------------------------------------------
        # Generating PDF Document
        self.pdf = initialize_pdf_doc()

        # Initialize dataframe for column/feature statistics
        self.master_columns_df = pd.DataFrame(index=data_df.columns)

        # --------------------------------------------------------------------
        # Null Values Analysis
        print('--- Null Values Analysis ---')

        self.master_columns_df, self.null_count_by_row_series = count_null_values(data_df, self.master_columns_df)

        # PDF Page 1: Summary of Null values information and unique values for numeric and non-numeric feature columns
        self.pdf = section_on_null_columns(self.pdf, data_df.shape[1], self.master_columns_df, self.null_count_by_row_series)
        print()

        # --------------------------------------------------------------------
        # Sort Numeric and Non-numeric/Categorical Columns
        print('--- Sorting Numeric and Non-numeric Columns / Unique Values ---')

        self.master_columns_df = sort_numeric_nonnumeric_columns(data_df, self.master_columns_df, self.target_col)

        self.pdf = section_on_unique_values(self.pdf, self.master_columns_df)

        # # Removing any columns with only a single unique value.

        # ---
        self.master_columns_df = calc_column_summary_stats(data_df, self.master_columns_df, self.target_type)


        # --------------------------------------------------------------------
        # Target Column

        if self.target_col is not None:
            print('\n--- Target Column ---')

            # Insert code / function here for target column nulls, unique values, distribution
            target_col_notnull = data_df[self.target_col].dropna()
            target_num_null = len(data_df) - len(target_col_notnull)
            target_num_uniq = target_col_notnull.nunique()

            if pd.api.types.is_string_dtype(target_col_notnull):
                self.target_type = 'classification'
            elif target_num_uniq <= 10:
                self.target_type = 'classification'
            else:
                self.target_type = 'regression'

            self.pdf = section_on_target_column(self.pdf, self.target_col, self.target_type, target_num_null,
                                                target_num_uniq)

        # Save PDF document to current working directory:
        if output:
            custom_filename = self.report_prefix + '_Initial'
            save_pdf_doc(self.pdf, custom_filename=custom_filename, timestamp=timestamp)

    def run_full_eda(self, data_df, run_collinear=True, generate_plots=True, plot_style='scatterdense',
                     set_plot_order=None):
        """
        Run a comprehensive exploratory data analysis (EDA) on a given dataset.

        This function runs the following steps:
        - The initial EDA -- see "run_initial_eda"
        - Feature correlations analysis
        - EDA Plots

        Parameters
        ----------
        data_df : Pandas dataframe of shape (n_samples, n data columns)
            The data to be analyzed.

        run_collinear : boolean, optional (default=True)
            To save time, one can set this to False, just to get the
            correlations between each feature and the target variable.

        generate_plots : bool, optional (default=True)
            This option is here to just get the correlations first, without
            waiting for all the plots to be generated.

        """

        # Save the current timestamp for the report filename and the folder to
        # contain the PNG plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # The directory is created for the EDA plots:
        plots_folder = './{}_EDA_plots_{}'.format(self.report_prefix, timestamp)
        Path(plots_folder).mkdir()

        # if self.cols_to_drop is not None:
        #     data_df = data_df.drop(columns=self.cols_to_drop)

        # Run the initial EDA steps:
        self.run_initial_eda(data_df, output=False)
        
        plot_hist_target_col(data_df[self.target_col].dropna(), target_type=self.target_type, inline=True,
                             set_plot_order=set_plot_order)
        plot_hist_target_col(data_df[self.target_col].dropna(), target_type=self.target_type,
                             set_plot_order=set_plot_order, plots_folder=plots_folder)
        self.pdf = section_on_target_column_plot(self.pdf, plots_folder)
        
        print()

        # --------------------------------------------------------------------
        # Calculate feature correlations

        print('--- Feature Correlations ---')

        self.numeric_cols = self.master_columns_df.loc[
            self.master_columns_df["Column Type"] == 'numeric'].index.to_list()
        self.non_numeric_cols = self.master_columns_df.loc[
            self.master_columns_df["Column Type"] == 'non-numeric'].index.to_list()

        count_w = 1 + math.floor(math.log10(len(data_df)))
        nlen = 34 - count_w

        # Calculate correlations between each numeric feature and the target
        # variable:
        if len(self.numeric_cols) > 0:
            self.master_columns_df = calc_numeric_features_target_corr(data_df, self.numeric_cols, self.master_columns_df,
                                                                       self.target_col, self.target_type, rf_n_estimators='auto')
            
            print('Numeric Features Correlations Summary  (Top-6 Correlated Features with Target)')
            tmp_df = self.master_columns_df.loc[
                self.master_columns_df["Column Type"] == 'numeric'].sort_values(by=["Random Forest"], ascending=False).iloc[0:6]
            corrn = 'Pearson' if self.target_type == 'regression' else 'MutInfo'
            col = "Pearson" if self.target_type == 'regression' else "Mutual Info"
            print(f'|--------------------------------------------------------|')
            print(f'| Numeric Feature        Count non-Null  {corrn}  RFcorr |')
            print(f'|--------------------------------------------------------|')
            for jj in range(len(tmp_df)):
                feat_name = tmp_df.index[jj]
                print(f'| {feat_name if len(feat_name) <= nlen else feat_name[0:nlen-3] + "...":<{nlen}}  '
                      f'{tmp_df["Count not-Null"].iloc[jj]:>{count_w}.0f}    {tmp_df[col].iloc[jj]: .2f}    '
                      f'{tmp_df["Random Forest"].iloc[jj]:.2f}  |')
            print(f'|--------------------------------------------------------|\n')
        else:
            run_collinear = False
        
        # Calculate correlations between numeric features:
        if run_collinear:
            self.numeric_collinear_df, self.master_columns_df = calc_corr_numeric_features(
                data_df, self.numeric_cols, self.master_columns_df)

        # Calculate correlations between each categorical feature and the
        # target variable:
        if len(self.non_numeric_cols) > 0:
            self.master_columns_df = calc_nonnumeric_features_target_corr(data_df, self.non_numeric_cols, self.master_columns_df,
                                                                          self.target_col, self.target_type)
            
            print('Non-Numeric Features Correlations Summary  (Top-6 Correlated Features with Target)')
            sort_col = "RF_norm" if "RF_norm" in self.master_columns_df.columns else "Random Forest"
            tmp_df = self.master_columns_df.loc[
                self.master_columns_df["Column Type"] == 'non-numeric'].sort_values(by=[sort_col], ascending=False).iloc[0:6]
            print(f'|---------------------------------------------------------------------------------|')
            print(f'| Non-Numeric Feature    Count non-Null  Num Uniq  Mut Info  RFcorr  RFcorr(norm) |')
            print(f'|---------------------------------------------------------------------------------|')
            for jj in range(len(tmp_df)):
                feat_name = tmp_df.index[jj]
                print(f'| {feat_name if len(feat_name) <= nlen else feat_name[0:nlen-3] + "...":<{nlen}}  '
                      f'{tmp_df["Count not-Null"].iloc[jj]:>{count_w}.0f}       '
                      f'{tmp_df["Num Unique Values"].iloc[jj]:>3.0f}      {tmp_df["Mutual Info"].iloc[jj]:.2f}    '
                      f'{tmp_df["Random Forest"].iloc[jj]:.2f}          {tmp_df["RF_norm"].iloc[jj]:.2f}  |')
            print(f'|---------------------------------------------------------------------------------|\n')
        
        # The counts of NULL values should be integers:
        if self.target_col is not None:
            self.master_columns_df.loc[self.target_col, "Count not-Null"] = data_df[self.target_col].notna().sum()
        self.master_columns_df["Count not-Null"] = self.master_columns_df["Count not-Null"].astype('Int64')
        
        # --------------------------------------------------------------------
        # Generating PDF Document

        # PDF Page 1: Generated during the 'run_initial_eda' function

        # PDF Pages 2-3: Summary of numeric and non-numeric feature
        # correlations:

        # plot_ecdf(np.abs(self.numeric_df["Pearson"].values), data_label="Pearson", outfile=False)
        # plot_ecdf(
        #     self.numeric_df["Random Forest"], data_label="Random Forest", xlabel='Correlation Value',
        #     filename='numeric_columns_target_correlation_ecdf', overplot=True, outfile=True, plots_folder=plots_folder)

        if len(self.numeric_cols) > 0:
            rf_vals = self.master_columns_df.loc[
                self.master_columns_df["Column Type"] == 'numeric', "Random Forest"].values
            if self.target_type == 'regression':
                pearson_vals = self.master_columns_df.loc[
                    self.master_columns_df["Column Type"] == 'numeric', "Pearson"].values
                plot_hist(data_for_bins=np.abs(pearson_vals), label_bins='Pearson (abs)', data_for_line=rf_vals,
                          label_line="RF_corr", xlabel='Correlation Value', ylabel='Feature Count',
                          filename='numeric_columns_target_correlation_hist', plots_folder=plots_folder)
            else:
                plot_hist(data_for_bins=rf_vals, label_bins='RF Cohen-Kappa', xlabel='Correlation Value',
                          ylabel='Feature Count', filename='numeric_columns_target_correlation_hist',
                          plots_folder=plots_folder)

        if run_collinear:
            plot_hist(data_for_bins=np.abs(self.numeric_collinear_df["Pearson"].values), label_bins='Pearson (abs)',
                      data_for_line=self.numeric_collinear_df["Random Forest"].values, label_line="RF_corr",
                      xlabel='Correlation Value', ylabel='Count of Numeric Feature Pairs',
                      filename='numeric_columns_collinear_correlation_hist', plots_folder=plots_folder)

        if len(self.non_numeric_cols) > 0:
            rf_vals = self.master_columns_df.loc[
                self.master_columns_df["Column Type"] == 'non-numeric', "Random Forest"].values
            rf_corr_vals = self.master_columns_df.loc[
                self.master_columns_df["Column Type"] == 'non-numeric', "RF_norm"].values
            if self.target_type == 'regression':
                plot_hist(data_for_bins=rf_vals, label_bins='RF_corr', data_for_line=rf_corr_vals,
                          label_line="RF_corr (norm)", xlabel='Correlation Value', ylabel='Feature Count',
                          filename='non_numeric_columns_target_correlation_hist', plots_folder=plots_folder)
            else:
                plot_hist(data_for_bins=rf_vals, label_bins='RF_corr', xlabel='Correlation Value',
                          ylabel='Feature Count', filename='non_numeric_columns_target_correlation_hist',
                          plots_folder=plots_folder)
        
        self.pdf = section_on_feature_corr(self.pdf, self.master_columns_df, self.target_type,
                                           plots_folder=plots_folder)

        # --------------------------------------------------------------------
        # Generate EDA plots

        if generate_plots:
            # ----------------------------------
            # Generate plots of numeric features

            if len(self.numeric_cols) > 0:
                # Order the features by correlation with target variable, in
                # descending order:
                columns_list_ordered = self.master_columns_df.loc[
                    self.master_columns_df["Column Type"] == 'numeric'].sort_values(
                        by=["Random Forest"], ascending=False).index.to_list()

                # Generate plots of numeric features, and save them to the
                # timestamped directory defined above:
                plot_feature_values(data_df, columns_list_ordered, self.master_columns_df, target_col=self.target_col,
                                    numeric=True, plot_style=plot_style, target_type=self.target_type,
                                    set_plot_order=set_plot_order, inline=True)
                
                plot_feature_values(data_df, columns_list_ordered, self.master_columns_df, target_col=self.target_col,
                                    numeric=True, plot_style=plot_style, target_type=self.target_type,
                                    set_plot_order=set_plot_order, plots_folder=plots_folder)

                # Add the plots to the PDF:
                self.pdf = section_of_plots(self.pdf, columns_list_ordered, target_col=self.target_col, numeric=True,
                                            plots_folder=plots_folder)

            # ----------------------------------
            # Generate plots of non-numeric features

            if len(self.non_numeric_cols) > 0:
                # Order the features by correlation with target variable, in
                # descending order:
                columns_list_ordered = self.master_columns_df.loc[
                    self.master_columns_df["Column Type"] == 'non-numeric'].sort_values(
                        by=["RF_norm"], ascending=False).index.to_list()

                # Generate plots of non-numeric features, and save them to the
                # timestamped directory defined above:
                plot_feature_values(data_df, columns_list_ordered, self.master_columns_df, target_col=self.target_col,
                                    numeric=False, plot_style=plot_style, target_type=self.target_type,
                                    set_plot_order=set_plot_order, inline=True)
                
                plot_feature_values(data_df, columns_list_ordered, self.master_columns_df, target_col=self.target_col,
                                    numeric=False, plot_style=plot_style, target_type=self.target_type,
                                    set_plot_order=set_plot_order, plots_folder=plots_folder)
                
                # Add the plots to the PDF:
                self.pdf = section_of_plots(self.pdf, columns_list_ordered, target_col=self.target_col, numeric=False,
                                            plots_folder=plots_folder)

        print('\n--- Files Output ---')
        # Save PDF document to current working directory:
        save_pdf_doc(self.pdf, custom_filename=self.report_prefix, timestamp=timestamp)
        print(f'All PNG files can be found in {plots_folder}.\n')

