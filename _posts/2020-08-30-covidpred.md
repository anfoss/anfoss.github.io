---
title: 'COVID prediction using Random forest'
date: 2020-08-30
permalink: /posts/2020/08/covidpred/
tags:
  - Machine learning
  - Proteomics
  - python3
  - covid
---


## Machine learning for proteomics: an easy introduction
In a lot of proteomics publications recently, machine learning algorithms are used to perform a variety of tasks such as sample classification, image segmentation or prediction of important features in a set of samples.

In this series I want to explore a bit how to employ machine learning in omics/proteomics and in general some good do's and don't in machine learning applications, plus providing some Python3 code to exemplify some of the ideas.
Only prerequisite is basic understanding of Python. I will drop explanation of things which I reckon be important but feel free to reach out for curiosities or similar

### Case study using random forest to predict COVID19 severity

Random forest is one of the most basic learning algorithm around the block and the easiest to apply. For a detailed explanation, there are several resources such as [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), but here I will focus on more hands-on tutorial and explain things I think are important as we go.
One of the major application where random forest is employed in proteomics paper is scoring of proteins across samples and sample classification.
Random forests can be conceptualized as a hierarchical stack of decision trees.

![](/images/rf_clf.png)

A decision tree boils down to a series of questions which allows to uniquely separate a group of samples from others.
So to calculate which features (i.e petal length or width) contributes more to classification, we can just observe how many misclassified samples are left after every decision.

This concept is known as Gini Impurity and is related to how pure a leaf is (how many samples of only one class are present) compared to the remaining ones.
Features leading to purer leaves are more important in the overall classification compared to other.

 __So we can retrieve how important a feature is and this will tell us about important proteins/analytes in our sample__

For this example I will use the COVID data from a recent Cell paper where a random forest classifier was used to classify severe and non-severe COVID19 patients based on metabolites and proteins.
All data is available (here)[https://www.cell.com/cell/fulltext/S0092-8674(20)30627-9#supplementaryMaterial].
Our goals are:
- Classify patients in severe and non-severe COVID19
- Identify features which are changing the most between severe and non-severe COVID
So let's start coding!


#### Import packages and prepare training and test datasets

We import and preprocess the data. Luckily for us, in the publication, the data is already separated in test and training set.
Every ML model needs to be trained on a set of data and then the generalization (i.e how good the model learned our data) capabilities are tested on a independent datasets.

If test data is not available usually the training data is split in a 0.75:0.25 training to test ratio or a 0.66/0.33 if there are sufficient data points.

In the publication the data is already merged, so we only need to get positive (severe covid-19) and negative (non-severe covid-19) and then assign it to a label.
For this, we will use supplementary table 1 which has the patient informations


```python
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


train = pd.read_excel('mmc3.xlsx', sheet_name='Prot_and_meta_matrix')
test = pd.read_excel('mmc4.xlsx', sheet_name='Prot_and_meta_matrix')
print(train.shape, test.shape)

```

    (1639, 33) (1590, 12)


As seen there is a different number of features in training (1639) and test (1590). While for some algorithms this doesn't matter, RF needs same number of features. So we will quickly fix it


```python
train = train[train['Proteins/Metabolites'].isin(test['Proteins/Metabolites'])]
test = test[test['Proteins/Metabolites'].isin(train['Proteins/Metabolites'])]
train.shape[0] == test.shape[0]
```




    True




```python
def target_features_split(info, df, sheet_name):
    """
    utility function to preprocess data and retrieve target (labels) and features (numerical values)
    """
    df.fillna(0, inplace=True)
    ids = df[['Proteins/Metabolites', 'Gene Symbol']]
    df.drop(['Proteins/Metabolites', 'Gene Symbol'], axis=1, inplace=True)
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    # needs to have features as columns and samples as row (i.e wide format)
    df = df.T
    df['class'] = df.index.map(info)
    return df.drop('class', axis=1).values, df['class'].values


# we need to have positive and negative information (i.e severe and )
info = pd.read_excel('mmc1.xlsx', index_col=0, sheet_name='Clinical_information')
info = info[info['Group d'].isin([2,3])]
info['class'] = [0 if x==2 else 1 for x in list(info['Group d'])]
info = dict(zip(info.index, info['class']))

# get training and test data in format for ml
Xtrain,ytrain = target_features_split(info, train, sheet_name='Prot_and_meta_matrix')
Xtest, ytest = target_features_split(info, test, sheet_name='Prot_and_meta_matrix')
```

#### Default random forest model

We can start by fitting a very simple model with all default parameters. For this, we initialized an empty classifier and then fit the data.
It is very important in machine learning to always use a seed (random_state in sklearn) to ensure reproducibility of the results.


```python
def evaluate(model, test_features, test_labels):
    """
    Evaluate model accuracy
    """
    predictions = model.predict(test_features)
    ov = np.equal(test_labels, predictions)
    tp = ov[np.where(ov ==True)].shape[0]
    return tp/predictions.shape[0]

# now we train on the features from the training set
clf_rf_default = RandomForestClassifier(random_state=42)
clf_rf_default.fit(Xtrain, ytrain)
print(evaluate(clf_rf_default, Xtest, ytest))
```

    0.7


Here we can see we got 70% recall and made two mistakes in classification where we predicted non severe covid (i.e 0) instead of severe covid (1).

Let's see if we can improve this by doing some parameter optimization. For this we will use a random grid search search. We will generate a set of various parameters and test random combinations to train our model.
Alternatively GridSearchCV can be used, where instead of using random combinations, all possible ones are tested.

RandomizedSearchCV and GridSearchCV use what is known as __cross validation__ or CV, which is a machine learning technique for model training where the data is splitted into equal parts (fold) and then all but one folds are used to train the model and the last one to predict. In this way a more robust estimation of model performance can obtained, but __the final model should be train on all available data which usually yields the best performance__.
For a more in depth explanation of cross validation, (here)[https://scikit-learn.org/stable/modules/cross_validation.html] there is an excellent introduction.


```python
def random_search():
    # Number of trees in random forest
    n_estimators = [250, 1000, 5000]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(20, 400, num = 40)]

    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']
    # % samples required to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    # Minimum % samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid

# initialize parameter space
grid = random_search()
clf = RandomForestClassifier(random_state=0)


# try 5 fold CV on 100 combinations aka 500 fits
rf_opt = RandomizedSearchCV(estimator = clf,
                       param_distributions = grid,
                       cv = 5,
                       n_iter=100,
                       verbose=1,
                       n_jobs = -1,
                       scoring='roc_auc')

# Fit the random search model
rf_opt_grid=rf_opt.fit(Xtrain, ytrain)

# retrieve best performing model
clf_rf_opt = rf_opt.best_estimator_
evaluate(clf_rf_opt, Xtest, ytest)
'Performance increase by {}%'.format(evaluate(clf_rf_opt, Xtest, ytest) - evaluate(clf_rf_default, Xtest, ytest))
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.6min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  6.3min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 13.1min
    [Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed: 14.8min finished





    'Performance increase by 0.10000000000000009%'



We can manually inspect the performance at every combination of parameters directly and compare it with the reported one from the paper (AUC 0.957) as we used the same score ('roc_auc') in GridSearchCV.


```python
res = pd.DataFrame.from_dict(rf_opt_grid.cv_results_)
res['mean_test_score'].max()
```




    0.9555555555555555



So the performance increased by just tuning the model.
We also achieved almost the same AUC (0.955) vs 0.957 reported in the paper so we manage to reproduce the classification. We could go further and double check which patient wasn't classified correctly, but let's take a break and continue in the next post!
Now, in the next part we will look into feature importance to figure out which proteins and metabolites allowed us to separate the samples.
We will also go in depth into some more tune-up and the problem of overfitting and the tradeoff bias/variance


```python

```
