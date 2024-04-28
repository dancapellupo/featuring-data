
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
[insert image of DF]

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


This is a simple example package. You can use
[GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
