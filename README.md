
# Featuring Data: Exploratory Data Analysis (EDA) and Feature Selection

Featuring Data is a Python library that builds on the well-known Pandas,
matplotlib, and scikit-learn libraries to provide an easy starting point for
EDA and feature selection on any structured dataset that is in the form of a
Pandas dataframe.

The two main parts of this library are the `FeaturesEDA` and the
`FeatureSelector` classes. Both classes provide easy options to create EDA
plots and a full PDF report in two lines of code.

## Installation and Dependencies

The Featuring Data library requires Python 3+, numpy , Pandas , matplotlib ,
seaborn, and scikit-learn.

The latest stable release (and required dependencies) can be installed from
PyPI:

[code here]

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
    swarm plot is generated. If there are more than 1,000 data points,
    then only a random selection of 1,000 points are plotted on the
    swarm plot (but the box plot is calculated based on all points).
  - For typical numeric features, a standard scatter plot is generated.

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
performance stay fairly constant, until you go down to about 20 features
(out of ~100 original features). The two colors represent two different
train/validation data splits.*
