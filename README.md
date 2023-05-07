# README for Breast Cancer Classification
This repository presents a codebase for analyzing the Breast Cancer dataset from the sklearn library and developing a classification model to predict the malignancy status of a tumor. According to the problems, this project used six classification algorithms, namely Decision Tree, Random Forest, XGBoost, SVC, KNN, and Logistic Regression, were employed to evaluate the dataset's performance. Specifically, the models were trained using different ratios of training data, and tested by accurcy score, recall, precision and f1 score. Finallt, I use 0.1 testing size and 0.9 training size to find the top three important features for each classification algorithm were determined. 

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

``` 
instances = len (cancer.data)
features = len (cancer.feature_names)
targets = len(cancer.target_names)

print("Number of instances: ", instances)
print("Number of features: ", features)
print("Number of targets(result): ", targets)
``` 

The number of instances (data points) in the dataset <br>
The number of features (independent variables) in the dataset <br>
The number of targets (dependent variables) in the dataset <br>

Additionally, it checks whether the dataset is imbalanced by counting the number of samples for each class and plotting the results using the matplotlib library.
``` 
n_benign = len(cancer.target[cancer.target == 0])
n_malignant = len(cancer.target[cancer.target == 1])
labels = ["Benign", "Malignant"]
values = [n_benign, n_malignant]
plt.bar(labels, values)
plt.title("Breast Cancer Dataset")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.show()
```
![](https://drive.google.com/uc?export=view&id=1B4DV-XfOzsRPufdKNg3SqXVWKOHkwDam)


## Model development and evaluation
The code in main.ipynb builds and evaluates a classification model using the Breast Cancer dataset. It splits the dataset into training and testing sets using the  `train_test_split() ` from the sklearn.model_selection module. It then creates a classification model using the sklearn library and fits it to the training data.

The `ratio_perform()` and `plot_ratio_perform()` functions evaluate the performance of the classification model for different train-test ratios, calculate the accuracy, recall, precision, and f1 score of the model on the training and testing sets, and plot the results. These functions are used to find the optimal train-test ratio for the model.
```
def ratio_perform(model, X, y, ratio=100):
  ratiovalues = [i for i in range(5, ratio, 5)]
  train_scores = []
  test_scores = []
  for i in ratiovalues:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i/100, random_state=2023)    
    clf = model
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train) 
    train_acc = accuracy_score(y_pred_train, y_train)
    
    y_pred_test = clf.predict(X_test) 
    test_acc = accuracy_score(y_pred_test, y_test)
    
    train_report = classification_report(y_train, y_pred_train)
    test_report = classification_report(y_test, y_pred_test)
    
    print('>%d, train:\n%s\ntest:\n%s' % (i, train_report, test_report))

    train_scores.append(train_acc)
    test_scores.append(test_acc)
```

```
def plot_ratio_perform(model, X, y, ratio=100):
  ratiovalues = [i for i in range(5, ratio, 5)]
  train_scores, test_scores = [], []
  train_recall, test_recall = [], []
  train_precision, test_precision = [], []
  train_f1, test_f1 = [], []
  
  for i in ratiovalues:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i/100, random_state=2023)
    clf = model
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_scores.append(accuracy_score(y_train, y_train_pred))
    test_scores.append(accuracy_score(y_test, y_test_pred))
    train_recall.append(recall_score(y_train, y_train_pred))
    test_recall.append(recall_score(y_test, y_test_pred))
    train_precision.append(precision_score(y_train, y_train_pred))
    test_precision.append(precision_score(y_test, y_test_pred))
    train_f1.append(f1_score(y_train, y_train_pred))
    test_f1.append(f1_score(y_test, y_test_pred))

  max_acc, max_acc_ratio = max((val, ratio) for ratio, val in zip(ratiovalues, test_scores))
  max_rec, max_rec_ratio = max((val, ratio) for ratio, val in zip(ratiovalues, test_recall))
  max_prec, max_prec_ratio = max((val, ratio) for ratio, val in zip(ratiovalues, test_precision))
  max_f1, max_f1_ratio = max((val, ratio) for ratio, val in zip(ratiovalues, test_f1))

  print(f"Max accuracy: {max_acc:.3f}, at test_size {max_acc_ratio/100}")
  print(f"Max recall: {max_rec:.3f}, at test_size {max_rec_ratio/100}")
  print(f"Max precision: {max_prec:.3f}, at test_size {max_prec_ratio/100}")
  print(f"Max F1-score: {max_f1:.3f}, at test_size {max_f1_ratio/100}")

  fig, axs = plt.subplots(2, 2, figsize=(10, 8))

  axs[0, 0].plot(ratiovalues, train_scores, '-o', label='Train')
  axs[0, 0].plot(ratiovalues, test_scores, '-o', label='Test')
  axs[0, 0].set_title('Accuracy')
  axs[0, 0].legend()

  axs[0, 1].plot(ratiovalues, train_recall, '-o', label='Train')
  axs[0, 1].plot(ratiovalues, test_recall, '-o', label='Test')
  axs[0, 1].set_title('Recall')
  axs[0, 1].legend()

  axs[1, 0].plot(ratiovalues, train_precision, '-o', label='Train')
  axs[1, 0].plot(ratiovalues, test_precision, '-o', label='Test')
  axs[1, 0].set_title('Precision')
  axs[1, 0].legend()

  axs[1, 1].plot(ratiovalues, train_f1, '-o', label='Train')
  axs[1, 1].plot(ratiovalues, test_f1, '-o', label='Test')
  axs[1, 1].set_title('F1-score')
  axs[1, 1].legend()

  plt.tight_layout()
  plt.show()
```

## Find the top 3 features in each models
Because I want to check whether one has breast cancer, I use precision to identify how many ratio is the best and the result is that trainning size = 0.9 and testing size = 0.1. <br>
To find the top 3 feaures, creat `plot_feature_importances` functon to plot the feature by `model.feature_importances_`
```
def plot_feature_importances(model):
    n_features = cancer.data.shape[1]
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.barh(np.arange(n_features), importances[indices], align='center')
    feature_names = np.array(cancer.feature_names)
    plt.yticks(np.arange(n_features), feature_names[indices])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()
```
However, SVC, KNN and Logistic Regression can't use `plot_feature_importances`. Because it doesn't have `model.feature_importances_` fuction. For SVC and Logistic Regression, I use `permutation_importance` from sklearn.inspection. 
```
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=2023, n_jobs=-1)

fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = result.importances_mean.argsort()
ax.barh(range(X.shape[1]), result.importances_mean[sorted_idx])
ax.set_yticks(range(X.shape[1]))
ax.set_yticklabels(cancer.feature_names[sorted_idx])
ax.set_xlabel('Permutation Importance')
plt.show()
```
For KNN, I use `mutual_info_classif` from sklearn.feature_selection. 
```
mutual_info = mutual_info_classif(X_train, y_train)

n_features = X.shape[1]
indices = np.argsort(mutual_info)
plt.barh(np.arange(n_features), mutual_info[indices], align='center')
feature_names = np.array(cancer.feature_names)
plt.yticks(np.arange(n_features), feature_names[indices])
plt.xlabel("Mutual Information")
plt.ylabel("Feature")
plt.ylim(-1, n_features)
plt.show()
```
