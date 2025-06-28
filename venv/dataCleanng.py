import pandas as pd
import numpy as np
import matplotlib as mlt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler

df=pd.read_csv("StudantPer.csv")

lable_encoder=LabelEncoder()
scale=StandardScaler()

df=df.drop_duplicates()


# Handle Missing Values
most_common=df["Parent Education"].mode()[0]
df["Parent Education"].fillna(most_common,inplace=True)


# 2. Normalized Text Columns

df["Gender"]=df["Gender"].str.lower()
df["Parent Education"]=df["Parent Education"].str.lower()
df["Region"]=df["Region"].str.lower()
df["Tutoring"]=df["Tutoring"].str.lower()





# Covert Data Into numeric
catorracl_data=["Gender","Tutoring","Region","Parent Education"]

for i in catorracl_data:
    df[i]=lable_encoder.fit_transform(df[i])



# Scaling Data
# df["Attendance"]=df["Attendance"].astype("")
numeric_feature=["HoursStudied/Week","Attendance"]
# print(df[numeric_feature].dtypes)
df[numeric_feature]=scale.fit_transform(df[numeric_feature])
print(df.head())






