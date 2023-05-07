# README for Breast Cancer Classification Project
This repository presents a codebase for analyzing the Breast Cancer dataset from the sklearn library and developing a classification model to predict the malignancy status of a tumor. According to the problems, this project used six classification algorithms, namely Decision Tree, Random Forest, XGBoost, SVC, KNN, and Logistic Regression, were employed to evaluate the dataset's performance. Specifically, the models were trained using different ratios of training data, and tested by accurcy score, recall, precision and f1 score. Finallt, I use 0.1 testing size and 0.9 training size to find the top three important features for each classification algorithm were determined. 

## Table of Contents
- [Description](#Description)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)

## Description
This code evaluates a classification model on the Breast Cancer Wisconsin dataset. The dataset is loaded using the `load_breast_cancer` function from scikit-learn. The following metrics are computed:
- Accuracy score
- Recall score
- F1 score
- Precision score
- Classification report
The evaluation is performed using the `train_test_split` function from scikit-learn to split the data into training and testing sets.

### Libraries Used
- scikit-learn
- numpy
- pandas
- matplotlib
- 
### Functions Imported
- accuracy_score
- recall_score
- f1_score
- precision_score
- classification_report
- tree
- RandomForestClassifier
- XGBClassifier
- svm
- neighbors
- LogisticRegression
- permutation_importance
- mutual_info_classif

## Background

### Any optional sections

## Install

This module depends upon a knowledge of [Markdown]().

```
```

### Any optional sections

## Usage

```
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.

### Any optional sections

## API

### Any optional sections

## More optional sections

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

### Any optional sections



