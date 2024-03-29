
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


def section_on_null_columns(pdf, num_features, null_cols_df, null_count_by_row_series):
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Null Values", ln=1)

    pdf.ln(2)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Null Values by Columns/Features", ln=1)

    pdf.set_font('Arial', '', 12)
    # TODO: Indicate also if target_col has nulls
    output_txt = "Out of {} total feature columns, there are {} columns with at least 1 null value.".format(
        num_features, len(null_cols_df))
    print(output_txt)
    pdf.cell(w=0, h=10, txt=output_txt, ln=1)

    pdf.ln(3)

    if len(null_cols_df) == 0:
        return pdf

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt=null_cols_df.columns[0], border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt=null_cols_df.columns[1], border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt=null_cols_df.columns[2], border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(8, len(null_cols_df))):
        pdf.cell(w=60, h=10, txt=null_cols_df["Feature"].iloc[ii], border=1, ln=0, align='L')
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


def section_on_unique_values(pdf, numeric_cols, non_numeric_cols, numeric_uniq_vals_df, non_numeric_uniq_vals_df,
                             single_value_cols_numeric_df=None, numeric_cols_to_cat_df=None,
                             single_value_cols_nonnumeric_df=None, nonnumeric_uniq_vals_thresh=5):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Numeric vs Non-Numeric Features and Unique Values Count", ln=1)

    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10,
             txt="Out of {} total feature columns, there are {} numeric columns and {} non-numeric columns.".format(
                 len(numeric_cols)+len(non_numeric_cols), len(numeric_cols), len(non_numeric_cols)),
             ln=1)

    pdf.ln(3)

    if ((single_value_cols_numeric_df is not None and len(single_value_cols_numeric_df) > 0) or
            (numeric_cols_to_cat_df is not None and len(numeric_cols_to_cat_df) > 0)):

        if single_value_cols_numeric_df is not None and len(single_value_cols_numeric_df) > 0:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(w=0, h=10,
                     txt="There are {} numeric columns with just a single value and will be removed.".format(
                         len(single_value_cols_numeric_df)), ln=1)
            pdf.ln(4)

        if numeric_cols_to_cat_df is not None and len(numeric_cols_to_cat_df) > 0:
            pdf.set_font('Arial', 'B', 12)
            output_txt = 'There are {} numeric columns that will be switched to categorical.'.format(
                len(numeric_cols_to_cat_df))
            print(output_txt)
            pdf.cell(w=0, h=10, txt=output_txt, ln=1)
            pdf.ln(4)

    else:
        # Table Header
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(w=60, h=10, txt='Numeric Feature', border=1, ln=0, align='C')
        pdf.cell(w=42, h=10, txt=numeric_uniq_vals_df.columns[1], border=1, ln=1, align='C')

        # Table Contents
        pdf.set_font('Arial', '', 12)
        for ii in range(0, min(8, len(numeric_uniq_vals_df))):
            pdf.cell(w=60, h=10,
                     txt=numeric_uniq_vals_df["Feature"].iloc[ii],
                     border=1, ln=0, align='L')
            pdf.cell(w=42, h=10,
                     txt=numeric_uniq_vals_df["Num Unique Values"].iloc[ii].astype(str),
                     border=1, ln=1, align='R')

        if len(numeric_uniq_vals_df) > 5:
            pdf.cell(w=0, h=10,
                     txt="There are an additional {} numeric feature columns with 10 or fewer unique values.".format(
                         len(numeric_uniq_vals_df) - 5),
                     ln=1)

    pdf.ln(4)

    if single_value_cols_nonnumeric_df is not None and len(single_value_cols_nonnumeric_df) > 0:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(w=0, h=10,
                 txt="There are {} non-numeric columns with just a single value and will be removed.".format(
                     len(single_value_cols_nonnumeric_df)), ln=1)
        pdf.ln(4)

    # Table Header
    non_numeric_uniq_vals_df_tmp = non_numeric_uniq_vals_df.loc[
        non_numeric_uniq_vals_df["Num Unique Values"] > nonnumeric_uniq_vals_thresh]

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Non-Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=42, h=10, txt=non_numeric_uniq_vals_df_tmp.columns[1], border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(8, len(non_numeric_uniq_vals_df_tmp))):
        pdf.cell(w=60, h=10,
                 txt=non_numeric_uniq_vals_df_tmp["Feature"].iloc[ii],
                 border=1, ln=0, align='L')
        pdf.cell(w=42, h=10,
                 txt=non_numeric_uniq_vals_df_tmp["Num Unique Values"].iloc[ii].astype(str),
                 border=1, ln=1, align='R')

    if len(non_numeric_uniq_vals_df_tmp) > 5:
        pdf.cell(w=0, h=10,
                 txt="There are an additional {} non-numeric feature columns with more than {} unique values.".format(
                     len(non_numeric_uniq_vals_df_tmp) - 5, nonnumeric_uniq_vals_thresh), ln=1)

    return pdf


def section_on_unique_values_p2(pdf, numeric_cols, non_numeric_cols):

    pdf.ln(3)
    pdf.set_font('Arial', '', 12)
    output_txt = ("After the above adjustments, there are now {} feature columns, with {} numeric columns and {} "
                  "non-numeric/categorical columns.").format(
        len(numeric_cols)+len(non_numeric_cols), len(numeric_cols), len(non_numeric_cols))
    print(output_txt)
    pdf.write(5, output_txt)

    return pdf


def section_on_feature_corr(pdf, numeric_df, numeric_collinear_summary_df, non_numeric_df, plots_folder='./'):

    pdf.add_page()

    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Feature Correlations", ln=1)

    # ------------------------------------------------------------------------
    # Numeric feature correlations with Target variable

    pdf.ln(2)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations of Numeric Features with Target Variable", ln=1)

    pdf.ln(2)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt='Count non-Null', border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt='Pearson Corr', border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt='RF Corr', border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(8, len(numeric_df))):
        pdf.cell(w=60, h=10, txt=numeric_df.index[ii], border=1, ln=0, align='L')
        pdf.cell(w=35, h=10, txt=numeric_df["Count not-Null"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=35, h=10, txt=numeric_df["Pearson"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=35, h=10, txt=numeric_df["Random Forest"].iloc[ii].astype(str), border=1, ln=1, align='R')

    pdf.ln(6)
    # pdf.image('{}/numeric_columns_target_correlation_ecdf.png'.format(plots_folder),
    #           x=10, y=None, w=160, h=0, type='PNG')
    pdf.image('{}/numeric_columns_target_correlation_hist.png'.format(plots_folder),
              x=10, y=None, w=160, h=0, type='PNG')
    pdf.ln(2)

    # output_txt = ('The above plot shows what fraction of numeric features have a correlation value with the target'
    #               'variable above a certain value.')
    output_txt = ('The above plot shows a histogram of all numeric features and their correlation value with the '
                  'target variable.')
    pdf.write(5, output_txt)
    pdf.ln(7)

    # pdf.write(5, 'In other words, the ')

    # ------------------------------------------------------------------------
    # Correlations Between Numeric Features

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations between Numeric Features", ln=1)

    pdf.ln(2)

    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(w=44, h=6, txt='', border='LTR', ln=0, align='C')
    pdf.cell(w=24, h=6, txt='Avg Pearson', border='LTR', ln=0, align='C')
    pdf.cell(w=23, h=6, txt='Avg RF', border='LTR', ln=0, align='C')
    pdf.cell(w=26, h=6, txt='Feat with', border='LTR', ln=0, align='C')
    pdf.cell(w=23, h=6, txt='', border='LTR', ln=0, align='C')
    pdf.cell(w=26, h=6, txt='Feat with', border='LTR', ln=0, align='C')
    pdf.cell(w=24, h=6, txt='', border='LTR', ln=1, align='C')

    pdf.cell(w=44, h=6, txt='Numeric Feature', border='LBR', ln=0, align='C')
    pdf.cell(w=24, h=6, txt='Correlation', border='LBR', ln=0, align='C')
    pdf.cell(w=23, h=6, txt='Correlation', border='LBR', ln=0, align='C')
    pdf.cell(w=26, h=6, txt='Max P Corr', border='LBR', ln=0, align='C')
    pdf.cell(w=23, h=6, txt='Max P Corr', border='LBR', ln=0, align='C')
    pdf.cell(w=26, h=6, txt='Max RF Corr', border='LBR', ln=0, align='C')
    pdf.cell(w=24, h=6, txt='Max RF Corr', border='LBR', ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 10)
    for ii in range(0, min(8, len(numeric_collinear_summary_df))):
        pdf.cell(w=44, h=10, txt=numeric_collinear_summary_df.index[ii], border=1, ln=0, align='L')
        pdf.cell(w=24, h=10, txt=numeric_collinear_summary_df["Avg Pearson Corr"].iloc[ii].astype(str), border=1,
                 ln=0, align='R')
        pdf.cell(w=23, h=10, txt=numeric_collinear_summary_df["Avg RF Corr"].iloc[ii].astype(str), border=1,
                 ln=0, align='R')
        pdf.set_font('Arial', '', 8)
        pdf.cell(w=26, h=10, txt=numeric_collinear_summary_df["Max Pear Corr Feature"].iloc[ii], border=1,
                 ln=0, align='L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(w=23, h=10, txt=numeric_collinear_summary_df["Max Pear"].iloc[ii].astype(str), border=1,
                 ln=0, align='R')
        pdf.set_font('Arial', '', 8)
        pdf.cell(w=26, h=10, txt=numeric_collinear_summary_df["Max RF Corr Feature"].iloc[ii], border=1,
                 ln=0, align='L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(w=24, h=10, txt=numeric_collinear_summary_df["Max RF Corr"].iloc[ii].astype(str), border=1,
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
        pdf.cell(w=60, h=10,
                 txt=non_numeric_df.index[ii],
                 border=1, ln=0, align='L')
        pdf.cell(w=32, h=10,
                 txt=non_numeric_df["Count not-Null"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=28, h=10,
                 txt=non_numeric_df["Num Unique"].iloc[ii].astype(str),
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
                  x=10, y=None, w=180, h=0, type='PNG')

    return pdf

