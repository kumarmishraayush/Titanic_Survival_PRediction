#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[14]:


# uploading the csv file 
titanic_data=pd.read_csv("train.csv")


# In[15]:


titanic_data.head()


# In[5]:


titanic_data.shape


# In[6]:


titanic_data.info()


# In[7]:


titanic_data.isnull().sum()


# In[22]:


titanic_data =titanic_data.drop(columns = 'Cabin',axis = 1)
titanic_data.isnull().sum()


# In[21]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
titanic_data.isnull().sum()


# In[16]:


#finding the mode value of "Embarked " column
print(titanic_data['Embarked'].mode())


# In[17]:


print(titanic_data['Embarked'].mode()[0])


# In[18]:


#replacing the missing value from "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace = True)


# In[24]:


titanic_data.isnull().sum()


# In[25]:


#getting some statistical measures about data
titanic_data.describe()


# In[28]:


# findng the number of people not survived
titanic_data['Survived'].value_counts()


# In[29]:


sns.set()


# In[31]:


#making a count plot for "Survived" column
sns.countplot('Survived',data=titanic_data)


# In[32]:


#making a count plot for "Sex" column
sns.countplot('Sex',data=titanic_data)


# In[35]:


#number of surviver gender based
sns.countplot('Sex', hue = 'Survived',data = titanic_data)


# In[36]:


sns.countplot('Pclass',data=titanic_data)


# In[37]:


sns.countplot('Pclass', hue = 'Survived',data = titanic_data)


# In[39]:


# Enconding  the categorical columns
titanic_data['Sex'].value_counts()


# In[40]:


titanic_data['Embarked'].value_counts()


# In[42]:


# converting cateogorical columns 

titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace =  True)


# In[43]:


titanic_data.head()


# In[54]:


#separating features and target
# droping some cloumn
X=titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis = 1)
Y = titanic_data['Survived']


# In[55]:


print(X)
print(Y)


# In[56]:


#spiting the data into training date and Test data


# In[58]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[60]:


print(X.shape,X_train.shape,X_test.shape)


# In[61]:


model = LogisticRegression()


# In[62]:


#training the logistic regression model  with training data
model.fit(X_train,Y_train)


# In[64]:


# model evaluation 
#accuracy on training data
X_train_prediction = model.predict(X_train)


# In[65]:


print(X_train_prediction)


# In[66]:


training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print('Accuracy score of training data : ',training_data_accuracy)


# In[68]:


X_test_prediction = model.predict(X_test)


# In[69]:


print(X_test_prediction)


# In[71]:


test_data_accuracy = accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of testing data : ',test_data_accuracy)


# In[ ]:




