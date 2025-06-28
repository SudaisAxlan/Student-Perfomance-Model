import pandas as pd
import numpy as np
import matplotlib as mlt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt


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
print(df.head())



# EDA Step Start

# # Cooralation 
# plt.figure(figsize=(10, 6))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation Heatmap")

# # Check relationships between features and target

# sns.scatterplot(x="HoursStudied/Week", y="Exam_Score", data=df)
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


