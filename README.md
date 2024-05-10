
<img src="https://raw.githubusercontent.com/dancapellupo/featuring-data/main/tmp/featuring-data-logo-md.png"><br>

-------------------------------------------------------------------------------

# Featuring Data: Exploratory Data Analysis (EDA) and Feature Selection

Featuring Data is a Python library that builds on the well-known Pandas,
matplotlib, and scikit-learn libraries to provide an easy starting point for
EDA and feature selection on any structured dataset that is in the form of a
Pandas dataframe.

The two main parts of this library are the `FeaturesEDA` and the
`FeatureSelector` classes. Both classes provide easy options to create EDA
plots and a full PDF report in two lines of code.

## Installation and Dependencies

The Featuring Data library requires Python>=3.8, numpy, pandas, matplotlib,
seaborn, scikit-learn, and xgboost.

The latest stable release (and required dependencies) can be installed from
PyPI:

```
pip install featuring-data
```

## Why Install this Package?

How many times do you have a new dataset - maybe someone sends you a new
Excel or CSV file - and you just want to do a quick EDA and get a quick
sense of the dataset before proceeding?

Do you find that your Jupyter notebook gets very long and very messy, very
quickly, when going through the different columns of a pandas dataframe,
creating plots, identifying how many unique values a particular column has,
etc.?

This package allows you to do this fast, so you quickly have a strong
starting point for further exploration.

Plus, you get
[a nicely formatted PDF](https://github.com/dancapellupo/featuring-data/blob/6efc4b54c65f8a2f38b498a01f569ad25bf2eb23/examples/Housing_Ames_EDA_Report_20240503_164941.pdf),
with all the usual, important details of a dataset layed out, for future
reference as you continue to work with a dataset.

Another bonus is that in creating this package, I have carefully researched
different methods and metrics. So, you won't be getting just the usual
Pearson correlation metrics or standard scatterplots.

Beyond EDA, there is a function to aid in feature selection. Going beyond
some of the usual feature selection techniques, this function performs an
iterative xgboost training, starting with all features and removing the
least "important" feature one-by-one. A nicely formatted PDF with different
plots helps visualize what is going on during the training and can help
uncover which features are driving the results.

## Get Started Quickly

After installing the package, open
[this Jupyter notebook](https://github.com/dancapellupo/featuring-data/blob/6efc4b54c65f8a2f38b498a01f569ad25bf2eb23/examples/featuring_data_regression_example.ipynb)
for a regression example or
[this Jupyter notebook](https://github.com/dancapellupo/featuring-data/blob/6efc4b54c65f8a2f38b498a01f569ad25bf2eb23/examples/featuring_data_classification_example.ipynb)
for a classification example, and run with the provided dataset (or read in
your own CSV or pandas dataframe).

[This PDF](https://github.com/dancapellupo/featuring-data/blob/6efc4b54c65f8a2f38b498a01f569ad25bf2eb23/examples/Housing_Ames_EDA_Report_20240503_164941.pdf)
shows an example output of the `FeaturesEDA` functionality.

## FeaturesEDA: A comprehensive EDA in two lines of code

This class implements Exploratory Data Analysis (EDA) on an input dataset.

```python
eda = FeaturesEDA(report_prefix='Housing_Ames', target_col="SalePrice", cols_to_drop=["Id"])
eda.run_full_eda(train_dataframe, run_collinear=True, generate_plots=True)
```

The results of the EDA are available within your Jupyter Notebook
environment for further EDA and analysis, and a nicely formatted PDF
report is generated and saved in your current working directory - for easy
reference or sharing results with team members and even stakeholders.

```python
eda.master_columns_df.head(5)
```

![Housing Ames 'master_columns_df' dataframe](/tmp/housing_ames_master_columns_df_head5.png)
*This is a truncated example of the main dataframe output of the EDA class,
showing each column from the training dataset in the left-most column (the
index of this dataframe), the number of Nulls in that column, the type of data
in that column, the number of unique values, and information about correlation
between each column of the dataset and the target column.*

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
    scatter density plot (the points are artificially spread for visual
    purposes) overplotted is generated.
  - For typical numeric features, a scatter density plot is generated. A color
    scale indicates how many overlapping points there are in a given location.

![Example visualizations of continuous and discrete variables](/tmp/housing_ames_example_feature_plots.png)
*An example plot of a numeric/continuous variable versus a continuous target
(left; the sale price of a house in Ames), and a discrete/categorical variable
versus the same continuous target (right).*

## FeatureSelector: Feature selection by recursive model training

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
    - The following information is kept from every iteration:
        - the feature importance values of every feature at every iteration
        - performance metrics on both the training and validation set
        - the number of features
        - the features removed at the end of each iteration

![](/tmp/housing_ames_num_features_vs_MAE.png)
*This plot shows that as the number of features is reduced, the model
performance stays fairly constant, until you go down to about 20 features
(out of ~100 original features). The two colors represent two different
train/validation data splits.*

## Credits

[1] Inspiration for the density scatterplots comes primarily from this
[StackOverflow post](https://stackoverflow.com/a/64105308).

[2] The example data set for regression comes from this
[Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

[3] The example data set for classification comes from this
[Kaggle competition](https://www.kaggle.com/competitions/titanic/data).

