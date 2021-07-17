from tkinter import *
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree
    
#Import Data
df = pd.read_csv("datarfe.csv")
features = ["Weekend","Discrimination","Need_to_do_H.Chores","Leave_for_Child Care","Award_Work","Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced"]
labels = ["Prod"]

#Ordinal Encoding
ord_enc = OrdinalEncoder()
headers = ["Weekend","Discrimination","Need_to_do_H.Chores","Leave_for_Child Care","Award_Work","Care-taker_WhileWorking","SP_Supp","Using_DCC","ComfyLeave_ChildSick","MatPat_Leave","Flex_Whours","Eval_WorkProduced","Prod"]

df["Weekend"] = ord_enc.fit_transform(df[["Weekend"]])
df["SP_Supp"] = ord_enc.fit_transform(df[["SP_Supp"]])
df["Using_DCC"] = ord_enc.fit_transform(df[["Using_DCC"]])
df["Award_Work"] = ord_enc.fit_transform(df[["Award_Work"]])
df["Discrimination"] = ord_enc.fit_transform(df[["Discrimination"]])
df["Need_to_do_H.Chores"] = ord_enc.fit_transform(df[["Need_to_do_H.Chores"]])
df["Leave_for_Child Care"] = ord_enc.fit_transform(df[["Leave_for_Child Care"]])
df["Care-taker_WhileWorking"] = ord_enc.fit_transform(df[["Care-taker_WhileWorking"]])
df["ComfyLeave_ChildSick"] = ord_enc.fit_transform(df[["ComfyLeave_ChildSick"]])
df["MatPat_Leave"] = ord_enc.fit_transform(df[["MatPat_Leave"]])
df["Flex_Whours"] = ord_enc.fit_transform(df[["Flex_Whours"]])
df["Eval_WorkProduced"] = ord_enc.fit_transform(df[["Eval_WorkProduced"]])
df["Prod"] = ord_enc.fit_transform(df[["Prod"]])


#Test-Train Split
X = df[features]
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.30)

#RFE
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=8, step=1)
selector = selector.fit(X, y.values.ravel())
print(selector.support_)
print(selector.ranking_)

