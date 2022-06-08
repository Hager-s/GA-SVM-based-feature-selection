import random
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


Data = pd.read_csv("series_matrix.csv", index_col=0)
Data = np.transpose(Data)
Data.columns = [col.replace('tumor (1) vs healty (0)', 'Class') for col in Data.columns]

X = Data.drop('Class', axis=1)
y = Data['Class']


#split data
X_train, X_test, y_train, y_test=train_test_split(Data.drop(labels=['Class'], axis=1),
    Data['Class'], test_size=0.3, random_state=41)


# feature extraction
k_best = SelectKBest(score_func=f_classif, k = 1000)
# fit on train set
fit = k_best.fit(X_train, y_train)
# transform train set
univariate_features = fit.transform(X_train)


feature_names=list(X_train.columns.values)
mask = k_best.get_support()
new_features = [] # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)


def CreateNewDataFrame(ListOfGenes,DataSet):

    data_frame = pd.DataFrame()
    for i in (ListOfGenes):
        data_frame[i] = DataSet[i]

    data_frame.insert(len(ListOfGenes),"Class",DataSet['Class'])
    return data_frame

New_DF = CreateNewDataFrame(new_features, Data)
New_DF.to_csv(r'Modified_Series_Matrix.csv')
print(New_DF)

