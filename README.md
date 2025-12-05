<H3>ENTER YOUR NAME</H3>  ABISHEK P
<H3>ENTER YOUR REGISTER NO.</H3>  212224040212
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
~~~
#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#Load dataset
data = pd.read_csv("Churn_Modelling.csv")
print("First 5 rows:\n", data.head())

#Explore dataset
print("\nDataset Info:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nStatistical Summary:\n")
print(data.describe())

#Drop irrelevant columns
# RowNumber, CustomerId, and Surname don't help prediction
data = data.drop(['RowNumber','CustomerId','Surname'], axis=1)

#Encode categorical variables (Geography, Gender)
label = LabelEncoder()
data['Geography'] = label.fit_transform(data['Geography'])
data['Gender'] = label.fit_transform(data['Gender'])

print("\nAfter Encoding:\n", data.head())

#Separate features and target
X = data.drop('Exited', axis=1).values   # features
y = data['Exited'].values                # target

#Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("\nNormalized Features (first 5 rows):\n", X_scaled[:5])

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

~~~


## OUTPUT:

<img width="733" height="469" alt="image" src="https://github.com/user-attachments/assets/9e849686-7f5c-436e-9b39-6d47419deeaf" />


<img width="496" height="522" alt="image" src="https://github.com/user-attachments/assets/3d8256e7-6830-43b5-8612-e8618ba76014" />


<img width="265" height="381" alt="image" src="https://github.com/user-attachments/assets/2fa44f57-3819-4d6d-a17c-5fd407a7d157" />


<img width="769" height="684" alt="image" src="https://github.com/user-attachments/assets/a4121b6c-00f1-4d68-9754-920fc33d31bd" />


<img width="749" height="579" alt="image" src="https://github.com/user-attachments/assets/66aee7b5-c760-45ee-a144-8839706ab2aa" />


<img width="289" height="57" alt="image" src="https://github.com/user-attachments/assets/b1569337-dd8a-4a5d-808c-3635598e34b1" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


