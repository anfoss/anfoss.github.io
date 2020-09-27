import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def fix_headers(df):
    """
    retrieve first row then create new headers
    """
    # trick to get column names
    oldcol = list(df)
    newcol = list(df.iloc[0])
    newcol[:2] = oldcol[:2]
    df.columns = newcol
    df.drop(df.index[0], inplace=True)
    return df


def target_features_split(info, df, sheet_name):
    """
    utility function to preprocess data and retrieve target (labels) and features (numerical values)
    """
    df.fillna(0, inplace=True)
    ids = df[['Proteins/Metabolites', 'Gene Symbol']]
    df.drop(['Proteins/Metabolites', 'Gene Symbol'], axis=1, inplace=True)
    # df needs to have features as columns and samples as row (i.e wide format)
    df = df.T
    df['class'] = df.index.map(info)
    return df.drop('class', axis=1).values, df['class'].values


def evaluate(model, test_features, test_labels):
    """
    Evaluate model accuracy
    """
    predictions = model.predict(test_features)
    ov = np.equal(test_labels, predictions)
    tp = ov[np.where(ov ==True)].shape[0]
    return tp / predictions.shape[0]

train = pd.read_excel('mmc3.xlsx', sheet_name='Prot_and_meta_matrix')
test = pd.read_excel('mmc4.xlsx', sheet_name='Prot_and_meta_matrix')
train = train[train['Proteins/Metabolites'].isin(test['Proteins/Metabolites'])]
test = test[test['Proteins/Metabolites'].isin(train['Proteins/Metabolites'])]

# fix headers and sort
train = fix_headers(train).sort_values(['Proteins/Metabolites'])
test = fix_headers(test).sort_values(['Proteins/Metabolites'])

# we need to have positive and negative information (i.e severe and not)
info = pd.read_excel('mmc1.xlsx', index_col=0, sheet_name='Clinical_information')
info = info[info['Group d'].isin([2,3])]
info['class'] = [0 if x==2 else 1 for x in list(info['Group d'])]
info = dict(zip(info.index, info['class']))

# get training and test data in format for ml
Xtrain,ytrain = target_features_split(info, train, sheet_name='Prot_and_meta_matrix')
Xtest, ytest = target_features_split(info, test, sheet_name='Prot_and_meta_matrix')

# now we train on the features from the training set
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(Xtrain, ytrain)

df = pd.DataFrame(
    {'feature': list(train['Proteins/Metabolites']),
     'importance': rf_clf.feature_importances_})

# filter features without importance
df = df[df['importance'] > 0]
df.sort_values(['importance'], ascending=False, inplace=True)

# sns.barplot(x="importance", y="feature", data=top20, color="salmon", saturation=.5)
