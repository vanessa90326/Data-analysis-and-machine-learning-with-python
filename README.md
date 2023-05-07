# README for Breast Cancer Classification
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

## Dataset analysis
The code in main.ipynb performs an analysis of the Breast Cancer dataset. It imports the dataset using the load_breast_cancer() function from the sklearn.datasets module and prints some basic information about the dataset:

` instances = len (cancer.data)
features = len (cancer.feature_names)
targets = len(cancer.target_names)

print("Number of instances: ", instances)
print("Number of features: ", features)
print("Number of targets(result): ", targets)` 

The number of instances (data points) in the dataset
The number of features (independent variables) in the dataset
The number of targets (dependent variables) in the dataset

Additionally, it checks whether the dataset is imbalanced by counting the number of samples for each class and plotting the results using the matplotlib library.
``` n_benign = len(cancer.target[cancer.target == 0])
n_malignant = len(cancer.target[cancer.target == 1])
labels = ["Benign", "Malignant"]
values = [n_benign, n_malignant]
plt.bar(labels, values)
plt.title("Breast Cancer Dataset")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.show()```




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



