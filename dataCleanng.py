import pandas as pd
import numpy as np
import matplotlib as mlt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


df=pd.read_csv("StudantPer.csv")

lable_encoder=LabelEncoder()
scale=StandardScaler()

# Data Cleaning Step Start

df=df.drop_duplicates()
df=df.drop(columns="Gender")


# Handle Missing Values
most_common=df["Parent Education"].mode()[0]
df["Parent Education"].fillna(most_common,inplace=True)

# 2. Normalized Text Columns
df["Parent Education"]=df["Parent Education"].str.lower()
df["Region"]=df["Region"].str.lower()
df["Tutoring"]=df["Tutoring"].str.lower()

# Covert Data Into numeric
catorracl_data=["Tutoring","Region","Parent Education"]

for i in catorracl_data:
    df[i]=lable_encoder.fit_transform(df[i])

# Scaling Data
numeric_feature=["HoursStudied/Week","Attendance"]
df[numeric_feature]=scale.fit_transform(df[numeric_feature])
# print(df.head())



# EDA Step Start

# # Cooralation 
# plt.figure(figsize=(10, 6))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation Heatmap")

# # Check relationships between features and target

# sns.scatterplot(x="HoursStudied/Week", y="Exam_Score", data=df)
# plt.title("Study Hours vs Exam Score")
# sns.scatterplot(x="HoursStudied/Week", y="Parent Education", data=df)
# plt.title("Study Hours vs Exam Score")

# sns.scatterplot(x="Tutoring", y="Exam_Score", data=df)
# plt.title("Tutoring vs Exam Score")

# sns.scatterplot(x="Region", y="Exam_Score", data=df)
# plt.title("Region vs Exam Score")

# sns.scatterplot(x="Attendance", y="Exam_Score", data=df)
# plt.title("Attendance vs Exam Score")

# sns.scatterplot(x="Parent Education", y="Exam_Score", data=df)
# plt.title("Parent Education vs Exam Score")

# # Check Outliers
# sns.boxplot(df)

# sns.boxplot(x=df['HoursStudied/Week'])
# sns.boxplot(x=df['Tutoring'])
# sns.boxplot(x=df['Region'])
# sns.boxplot(x=df['Attendance'])
# sns.boxplot(x=df['Parent Education'])

# plt.title("Study Hours vs Exam Score")

#Understanding distributions 
# sns.histplot(df["HoursStudied/Week"], kde=True)
# plt.show()


# Feature Enginering Steps Start
# New Features

# print(df.head())
df["Study_faimlySupport"]=df["HoursStudied/Week"]* df["Parent Education"]
df["Study_Attend"] = df["HoursStudied/Week"] * df["Attendance"]
print("**************")
# print(df.head())


# Spliting Data
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]
x=df
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# Selection And Training Model

# Linear Model
linerModel=LinearRegression()
linerModel.fit(X,y)
y_pre=linerModel.predict(x_test)
print("Linear Reagaeartion R2 Score:", r2_score(y_test, y_pre))
print(" Linear Reagaeartion  MSE",mean_squared_error(y_test,y_pre))

print("******")
# Decision Tree Model
destion_treeModel=DecisionTreeRegressor()
destion_treeModel.fit(X,y)
destion_y_predt=destion_treeModel.predict(x_test)
print("Destion Tree R2 Score ",r2_score(y_test,destion_y_predt))
print("Destion Tree MSE Score ",mean_squared_error(y_test,destion_y_predt))

print("***********")
# RandomForest Model
randomForestModel=RandomForestRegressor()
randomForestModel.fit(X,y)
y_predt_RandomForest=randomForestModel.predict(x_test)
print("Randeom orest r2 Score",r2_score(y_test,y_predt_RandomForest))
print("Randrrom Foest r2 MSE",mean_squared_error(y_test,y_predt_RandomForest))


