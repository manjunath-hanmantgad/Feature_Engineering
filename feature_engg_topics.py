# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:01:54 2022

@author: Manjunath
"""

# List of Feature engineering techniques


''' 
@ Imputation 

Used in dealing with missing values.
There is not an optimum threshold for dropping but you can use 70% as an example value and try to drop the rows and columns which have missing values with higher than this threshold.

'''

threshold = 0.7
#Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]

# Numerical Imputation

''' The best imputation way is to use the medians of the columns. As the averages of the columns are sensitive to the outlier values, while medians are more solid in this respect.'''

#Filling all missing values with 0
data = data.fillna(0)
#Filling missing values with medians of the columns
data = data.fillna(data.median())


# Categorical Imputation
'''Replacing the missing values with the maximum occurred value in a column '''

#Max fill function for categorical columns
data['column_name'].fillna(data['column_name'].value_counts().idxmax(), inplace=True)


'''
@ Handling Outliers
'''

# Use standard deviation, and percentiles to calculate outliers
# Use Z score to calculate outlier with standard deviation

# Using percentiles

'''
Outlier Detection with Percentiles
Another mathematical method to detect outliers is to use percentiles. You can assume a certain percent of the value from the top or the bottom as an outlier. The key point is here to set the percentage value once again
'''


''' 
@ Binning

Binning can be applied on both categorical and numerical data.
The main motivation of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance. Every time you bin something, you sacrifice information and make your data more regularized.

For example, if your data size is 100,000 rows, it might be a good option to unite the labels with a count less than 100 to a new category like “Other”.
''' 


'''
@ Log Transform

1. helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
2. log transform normalizes the magnitude differences like that
3. decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
'''


'''
@ One-hot encoding

This method spreads the values in a column to multiple flag columns and assigns 0 or 1 to them.

a.) Categorical Column Grouping
b) using pivot table
c) apply a group by function after applying one-hot encoding

d) Numerical Column Grouping

'''

'''
@ Scaling

the algorithms based on distance calculations such as k-NN or k-Means need to have scaled continuous features as model input.

2 ways to apply scaling

1. Normalization 

Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. This transformation does not change the distribution of the feature and due to the decreased standard deviations, the effects of the outliers increases. Therefore, before normalization, it is recommended to handle the outliers.


2. Standardization

Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features is different, their range also would differ from each other. This reduces the effect of the outliers in the features.


'''

