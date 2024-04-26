
from fpdf import FPDF

import numpy as np


def initialize_pdf_doc():
    pdf = FPDF()

    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=0, h=10, txt="Feature Selection and EDA Report", ln=1)
    pdf.ln(2)

    return pdf


def save_pdf_doc(pdf, custom_filename='FeatureSelection', timestamp=''):
    pdf.output('./{}_EDA_Report_{}.pdf'.format(custom_filename, timestamp), 'F')
    print("File '{}_EDA_Report_{}.pdf' has been saved in current working directory.".format(custom_filename, timestamp))


def adjust_fontsize_for_feature_names(pdf, feature, box_width=60, start_fontsize=12):

    # sum(i.isupper() for i in a)
    feat_len_adj = len(feature) + np.floor(sum(map(str.isupper, feature)) / 2).astype(int)

    if box_width == 60:
        start_shrink = 27
        font_size_dict = {27: 11, 28: 11, 29: 11, 30: 10, 31: 10, 32: 10, 33: 10, 34: 9, 35: 9, 36: 9, 37: 9, 38: 8,
                          39: 8, 40: 8, 41: 7, 42: 7, 43: 7, 44: 7}
        start_cutoff = 45
    elif box_width == 44:
        start_shrink = 23
        font_size_dict = {23: 9, 24: 9, 25: 9, 26: 8, 27: 8, 28: 8, 29: 8, 30: 7, 31: 7, 32: 7, 33: 7}
        start_cutoff = 34
    elif box_width == 33:
        start_shrink = 23
        font_size_dict = {23: 7, 24: 7}
        start_cutoff = 25

    if feat_len_adj >= start_shrink:
        if feat_len_adj >= start_cutoff:
            feature = feature[0:start_cutoff - 3 - np.floor(
                sum(map(str.isupper, feature[0:start_cutoff-3])) / 3.5).astype(int)] + '...'
            font_size = 7
        else:
            font_size = font_size_dict[feat_len_adj]
        pdf.set_font('Arial', '', font_size)
    pdf.cell(w=box_width, h=10, txt=feature, border=1, ln=0, align='L')
    pdf.set_font('Arial', '', start_fontsize)

    return pdf


def section_on_null_columns(pdf, num_features, master_columns_df, null_count_by_row_series):

    null_cols_df = master_columns_df.loc[
        master_columns_df["Num of Nulls"] > 0, ["Num of Nulls", "Frac Null"]].sort_values(
            by=["Num of Nulls"], ascending=False)

    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Null Values", ln=1)

    pdf.ln(2)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Null Values by Columns/Features", ln=1)

    pdf.set_font('Arial', '', 12)
    output_txt = "Out of {} total data columns, there are {} columns with at least 1 null value.".format(
        num_features, len(null_cols_df))
    print(output_txt + '\n')
    pdf.cell(w=0, h=10, txt=output_txt, ln=1)

    pdf.ln(3)

    if len(null_cols_df) == 0:
        return pdf

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt="Feature", border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt=null_cols_df.columns[0], border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt=null_cols_df.columns[1], border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(8, len(null_cols_df))):
        pdf = adjust_fontsize_for_feature_names(pdf, null_cols_df.index[ii])
        pdf.cell(w=35, h=10, txt=null_cols_df["Num of Nulls"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=35, h=10, txt=null_cols_df["Frac Null"].iloc[ii].astype(str), border=1, ln=1, align='R')

    pdf.ln(6)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Null Values by Rows/Data Samples", ln=1)

    null_count_by_row = null_count_by_row_series.values
    xx = np.where(null_count_by_row > 0)[0]
    output_txt = 'Out of {} total rows/data samples, {} rows have at least one null value.'.format(
        null_count_by_row.size, xx.size)
    print(output_txt)
    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10, txt=output_txt, ln=1)

    # report how many rows have greater than 25% / 50% nulls
    for frac in [0.25, 0.50, 0.75, 1.]:
        xx = np.where(null_count_by_row > frac*num_features)[0]
        if xx.size > 0:
            output_txt = 'There are {} rows where at least {:.0f}% of the values are NULL.'.format(xx.size, frac*100)
            print(output_txt)
            pdf.cell(w=0, h=10, txt=output_txt, ln=1)
        else:
            break

    output_txt = 'The row with the most NULL values has {} NULLs.'.format(np.max(null_count_by_row))
    print(output_txt)
    pdf.cell(w=0, h=10, txt=output_txt, ln=1)

    return pdf


def section_on_unique_values(pdf, master_columns_df, nonnumeric_uniq_vals_thresh=5):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Numeric vs Non-Numeric Features and Unique Values Count", ln=1)

    num_numeric_cols = len(master_columns_df.loc[master_columns_df["Column Type (orig)"] == 'numeric'])
    num_nonnumeric_cols = len(master_columns_df.loc[master_columns_df["Column Type (orig)"] == 'non-numeric'])

    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10,
             txt="Out of {} total feature columns, there are {} numeric columns and {} non-numeric columns.".format(
                 num_numeric_cols+num_nonnumeric_cols, num_numeric_cols, num_nonnumeric_cols),
             ln=1)

    pdf.ln(3)

    # Table Header
    max_table_len = 8
    numeric_uniq_vals_df = master_columns_df.loc[
        (master_columns_df["Column Type (orig)"] == 'numeric') & (master_columns_df["Num Unique Values"] <= 10),
        ["Num Unique Values"]].sort_values(by=["Num Unique Values"])
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=42, h=10, txt=numeric_uniq_vals_df.columns[0], border=1, ln=1, align='C')

    pdf.set_font('Arial', '', 12)
    if len(numeric_uniq_vals_df) == 0:
        output_txt = 'There are no numeric columns with 10 or fewer unique values.'
        print(output_txt)
        pdf.cell(w=0, h=10, txt=output_txt, ln=1)
    
    else:
        # Table Contents
        for ii in range(0, min(max_table_len, len(numeric_uniq_vals_df))):
            pdf = adjust_fontsize_for_feature_names(pdf, numeric_uniq_vals_df.index[ii])
            pdf.cell(w=42, h=10,
                    txt=numeric_uniq_vals_df["Num Unique Values"].iloc[ii].astype(str),
                    border=1, ln=1, align='R')

    if len(numeric_uniq_vals_df) > max_table_len:
        pdf.cell(w=0, h=10,
                 txt="There are an additional {} numeric feature columns with 10 or fewer unique values.".format(
                     len(numeric_uniq_vals_df) - max_table_len),
                 ln=1)

    pdf.ln(4)

    if len(master_columns_df.loc[master_columns_df["Column Type (orig)"] == 'numeric', "Column Note"].dropna()) > 0:
        num_single_val = len(master_columns_df.loc[
            (master_columns_df["Column Type (orig)"] == 'numeric') & (master_columns_df["Column Note"] == 'remove')])
        
        if num_single_val > 0:
            pdf.set_font('Arial', 'B', 12)
            output_txt = "There are {} numeric columns with just a single value and will be removed.".format(
                num_single_val)
            print(output_txt)
            pdf.cell(w=0, h=10, txt=output_txt, ln=1)
            pdf.ln(6)

        num_switch = len(master_columns_df.loc[master_columns_df["Column Note"] == 'switch to non-numeric'])
        
        if num_switch > 0:
            pdf.set_font('Arial', 'B', 12)
            output_txt = 'There are {} numeric columns that will be switched to categorical.'.format(
                num_switch)
            print(output_txt)
            pdf.cell(w=0, h=10, txt=output_txt, ln=1)
            pdf.ln(6)

    # Table Header
    non_numeric_uniq_vals_df_tmp = master_columns_df.loc[
        (master_columns_df["Column Type (orig)"] == 'non-numeric') & (master_columns_df["Num Unique Values"] > nonnumeric_uniq_vals_thresh),
        ["Num Unique Values"]].sort_values(by=["Num Unique Values"], ascending=False)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Non-Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=42, h=10, txt=non_numeric_uniq_vals_df_tmp.columns[0], border=1, ln=1, align='C')

    pdf.set_font('Arial', '', 12)
    if len(non_numeric_uniq_vals_df_tmp) == 0:
        output_txt = 'There are no non-numeric columns with more than {} unique values.'.format(nonnumeric_uniq_vals_thresh)
        print(output_txt)
        pdf.cell(w=0, h=10, txt=output_txt, ln=1)

    else:
        # Table contents
        for ii in range(0, min(max_table_len, len(non_numeric_uniq_vals_df_tmp))):
            pdf = adjust_fontsize_for_feature_names(pdf, non_numeric_uniq_vals_df_tmp.index[ii])
            pdf.cell(w=42, h=10,
                    txt=non_numeric_uniq_vals_df_tmp["Num Unique Values"].iloc[ii].astype(str),
                    border=1, ln=1, align='R')

    if len(non_numeric_uniq_vals_df_tmp) > max_table_len:
        pdf.cell(w=0, h=10,
                 txt="There are an additional {} non-numeric feature columns with more than {} unique values.".format(
                     len(non_numeric_uniq_vals_df_tmp) - max_table_len, nonnumeric_uniq_vals_thresh), ln=1)

    pdf.ln(4)

    num_issues = len(master_columns_df.loc[master_columns_df["Column Type (orig)"] == 'non-numeric', "Column Note"].dropna())
    if num_issues > 0:
        num_single = len(master_columns_df.loc[
            (master_columns_df["Column Type (orig)"] == 'non-numeric') & (master_columns_df["Num Unique Values"] == 1)])
        pdf.set_font('Arial', 'B', 12)
        output_txt = "There are {} non-numeric columns with just a single value and will be removed.".format(
            num_single)
        print(output_txt)
        pdf.cell(w=0, h=10, txt=output_txt, ln=1)
        output_txt = "There are {} non-numeric columns with a very large number of unique values and will be removed.".format(
            num_issues-num_single)
        print(output_txt)
        pdf.cell(w=0, h=10, txt=output_txt, ln=1)
        pdf.ln(4)
    
    if len(master_columns_df["Column Note"].dropna()) > 0:
        pdf.ln(8)
        num_numeric_cols = len(master_columns_df.loc[master_columns_df["Column Type"] == 'numeric'])
        num_nonnumeric_cols = len(master_columns_df.loc[master_columns_df["Column Type"] == 'non-numeric'])
        pdf.set_font('Arial', '', 12)
        output_txt = ("After the above adjustments, there are now {} data columns, with {} numeric columns and {} "
                      "non-numeric/categorical columns.").format(
                          num_numeric_cols+num_nonnumeric_cols, num_numeric_cols, num_nonnumeric_cols)
        print(output_txt)
        pdf.write(5, output_txt)

    return pdf


def section_on_target_column(pdf, target_col, target_type, target_num_null, target_num_uniq):

    pdf.add_page()

    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Target Column", ln=1)

    pdf.ln(2)
    pdf.set_font('Arial', '', 12)
    output_txt = "For the chosen target column ('{}'), this appears to be a {} problem.".format(
        target_col, target_type)
    print(output_txt)
    pdf.write(5, output_txt)

    pdf.ln(6)
    output_txt = "The target column has {} null values and {} unique values.".format(
        target_num_null, target_num_uniq)
    print(output_txt)
    pdf.write(5, output_txt)

    pdf.ln(12)

    return pdf


def section_on_target_column_plot(pdf, plots_folder='./'):

    pdf.image('{}/target_data_distribution.png'.format(plots_folder), x=10, y=None, w=160, h=0, type='PNG')
    pdf.ln(2)

    output_txt = "The above plot shows the distribution of values in the target column."
    pdf.write(5, output_txt)

    return pdf


def section_on_feature_corr(pdf, master_columns_df, target_type='regression', plots_folder='./'):

    pdf.add_page()

    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Feature Correlations", ln=1)

    # ------------------------------------------------------------------------
    # Numeric feature correlations with Target variable

    pdf.ln(2)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations of Numeric Features with Target Variable", ln=1)

    pdf.ln(2)

    numeric_df = master_columns_df.loc[
        master_columns_df["Column Type"] == 'numeric'].sort_values(by=["Random Forest"], ascending=False)

    if len(numeric_df) == 0:
        pdf.cell(w=0, h=10, txt="** No numeric features in this dataset. **", align='C')

    else:
        # Table Header
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(w=60, h=10, txt='Numeric Feature', border=1, ln=0, align='C')
        pdf.cell(w=35, h=10, txt='Count non-Null', border=1, ln=0, align='C')
        if target_type == 'regression':
            pdf.cell(w=35, h=10, txt='Pearson Corr', border=1, ln=0, align='C')
        pdf.cell(w=35, h=10, txt='RF Corr', border=1, ln=1, align='C')

        # Table contents
        pdf.set_font('Arial', '', 12)
        for ii in range(0, min(8, len(numeric_df))):
            pdf = adjust_fontsize_for_feature_names(pdf, numeric_df.index[ii])
            pdf.cell(w=35, h=10, txt=numeric_df["Count not-Null"].iloc[ii].astype(str), border=1, ln=0, align='R')
            if target_type == 'regression':
                pdf.cell(w=35, h=10, txt=numeric_df["Pearson"].iloc[ii].astype(str), border=1, ln=0, align='R')
            pdf.cell(w=35, h=10, txt=numeric_df["Random Forest"].iloc[ii].astype(str), border=1, ln=1, align='R')

        pdf.ln(6)
        pdf.image('{}/numeric_columns_target_correlation_hist.png'.format(plots_folder),
                  x=10, y=None, w=160, h=0, type='PNG')
        pdf.ln(2)

        output_txt = ('The above plot shows a histogram of all numeric features and their correlation value with the '
                      'target variable.')
        pdf.write(5, output_txt)
        pdf.ln(7)

    # ------------------------------------------------------------------------
    # Correlations Between Numeric Features

    if "COLLIN Max RF Corr" in master_columns_df.columns:
        numeric_collinear_summary_df = master_columns_df.loc[
            master_columns_df["Column Type"] == 'numeric'].sort_values(by=["COLLIN Max RF Corr"], ascending=False)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(w=0, h=10, txt="Correlations between Numeric Features", ln=1)

        pdf.ln(2)

        # Table Header
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(w=44, h=6, txt='', border='LTR', ln=0, align='C')
        pdf.cell(w=23, h=6, txt='Avg Pearson', border='LTR', ln=0, align='C')
        pdf.cell(w=22, h=6, txt='Avg RF', border='LTR', ln=0, align='C')
        pdf.cell(w=33, h=6, txt='Feat with Max', border='LTR', ln=0, align='C')
        pdf.cell(w=18, h=6, txt='Max Pear', border='LTR', ln=0, align='C')
        pdf.cell(w=33, h=6, txt='Feat with Max', border='LTR', ln=0, align='C')
        pdf.cell(w=17, h=6, txt='Max RF', border='LTR', ln=1, align='C')

        pdf.cell(w=44, h=6, txt='Numeric Feature', border='LBR', ln=0, align='C')
        pdf.cell(w=23, h=6, txt='Correlation', border='LBR', ln=0, align='C')
        pdf.cell(w=22, h=6, txt='Correlation', border='LBR', ln=0, align='C')
        pdf.cell(w=33, h=6, txt='Pear Corr', border='LBR', ln=0, align='C')
        pdf.cell(w=18, h=6, txt='Corr', border='LBR', ln=0, align='C')
        pdf.cell(w=33, h=6, txt='RF Corr', border='LBR', ln=0, align='C')
        pdf.cell(w=17, h=6, txt='Corr', border='LBR', ln=1, align='C')

        # Table contents
        pdf.set_font('Arial', '', 10)
        for ii in range(0, min(8, len(numeric_collinear_summary_df))):
            pdf = adjust_fontsize_for_feature_names(
                pdf, numeric_collinear_summary_df.index[ii], box_width=44, start_fontsize=10)
            pdf.cell(w=23, h=10, txt=numeric_collinear_summary_df["COLLIN Avg Pearson Corr"].iloc[ii].astype(str), border=1,
                     ln=0, align='R')
            pdf.cell(w=22, h=10, txt=numeric_collinear_summary_df["COLLIN Avg RF Corr"].iloc[ii].astype(str), border=1,
                     ln=0, align='R')
            pdf.set_font('Arial', '', 8)
            pdf = adjust_fontsize_for_feature_names(
                pdf, numeric_collinear_summary_df["COLLIN Max Pear Corr Feature"].iloc[ii], box_width=33, start_fontsize=10)
            pdf.cell(w=18, h=10, txt=numeric_collinear_summary_df["COLLIN Max Pear"].iloc[ii].astype(str), border=1,
                     ln=0, align='R')
            pdf.set_font('Arial', '', 8)
            pdf = adjust_fontsize_for_feature_names(
                pdf, numeric_collinear_summary_df["COLLIN Max RF Corr Feature"].iloc[ii], box_width=33, start_fontsize=10)
            pdf.cell(w=17, h=10, txt=numeric_collinear_summary_df["COLLIN Max RF Corr"].iloc[ii].astype(str), border=1,
                     ln=1, align='R')

        pdf.ln(6)
        pdf.image('{}/numeric_columns_collinear_correlation_hist.png'.format(plots_folder),
                  x=15, y=None, w=160, h=0, type='PNG')
        pdf.ln(2)

        output_txt = ('The above plot shows a histogram of all unique pairs of numeric features and the correlation '
                      'between the two features of each the pair.')
        pdf.write(5, output_txt)
        pdf.ln(7)

    # ------------------------------------------------------------------------
    # Non-numeric Feature Correlations with Target Variable

    pdf.add_page()

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations of Non-Numeric Features with Target Variable", ln=1)

    pdf.ln(2)

    sort_col = "RF_norm" if "RF_norm" in master_columns_df.columns else "Random Forest"
    non_numeric_df = master_columns_df.loc[
        master_columns_df["Column Type"] == 'non-numeric'].sort_values(by=[sort_col], ascending=False)

    if len(non_numeric_df) == 0:
        pdf.cell(w=0, h=10, txt="** No non-numeric features in this dataset. **", align='C')
        return pdf

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Non-Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=32, h=10, txt='Count non-Null', border=1, ln=0, align='C')
    pdf.cell(w=28, h=10, txt='Num Unique', border=1, ln=0, align='C')
    pdf.cell(w=30, h=10, txt='RF Corr', border=1, ln=0, align='C')
    pdf.cell(w=30, h=10, txt='RF Corr (norm)', border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(8, len(non_numeric_df))):
        pdf = adjust_fontsize_for_feature_names(pdf, non_numeric_df.index[ii])
        pdf.cell(w=32, h=10,
                 txt=non_numeric_df["Count not-Null"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=28, h=10,
                 txt=non_numeric_df["Num Unique Values"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=30, h=10,
                 txt=non_numeric_df["Random Forest"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=30, h=10,
                 txt=non_numeric_df["RF_norm"].iloc[ii].astype(str),
                 border=1, ln=1, align='R')

    pdf.ln(6)
    pdf.image('{}/non_numeric_columns_target_correlation_hist.png'.format(plots_folder),
              x=15, y=None, w=160, h=0, type='PNG')
    pdf.ln(2)

    output_txt = ('The above plot shows a histogram of all non-numeric features and their correlation value with the '
                  'target variable.')
    pdf.write(5, output_txt)

    return pdf


def section_of_plots(pdf, columns_list, target_col, numeric=True, plots_folder='./plots'):

    pdf.add_page()
    pdf.set_font('Arial', 'B', 13)
    if numeric:
        pdf.cell(w=0, h=200, txt="Plots of Numeric Columns versus the Target Variable", ln=1, align='C')
    else:
        pdf.cell(w=0, h=200, txt="Plots of Non-Numeric Columns versus the Target Variable", ln=1, align='C')

    for jj, column in enumerate(columns_list):

        if (jj % 2) == 0:
            pdf.add_page()
        else:
            pdf.ln(4)

        # TODO: Double-check that file exists
        pdf.image('{}/{}_vs_{}.png'.format(plots_folder, column, target_col),
                  x=10, y=None, w=0, h=130, type='PNG')
        
        if jj == 150:
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(w=0, h=20, txt="Any additional plots are located in '{}'.".format(plots_folder), align='C')
            break

    return pdf

