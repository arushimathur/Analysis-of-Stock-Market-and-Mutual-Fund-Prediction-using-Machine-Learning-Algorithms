#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


datasource=r'd:\Stock\companiestock.xlsx'


# In[4]:


df=pd.read_excel(datasource)


# In[5]:


#tcriteria=df['Median']>0


# In[6]:


#print(df[tcriteria].tail())


# In[7]:


print(df.loc[:,['Median']].agg(['mean','median','std']))


# In[8]:


print(df.loc[:,['Daychg']].agg(['mean','median','std']))


# In[9]:


df1=df.copy()


# In[10]:


df1['Selected']=np.where(df['Median']>102.639999 ,1,0)


# In[11]:


df1.head()


# In[12]:


datasource1=r'd:\Stock\companiesstock1.xlsx'


# In[13]:


#df1.to_excel(datasource1)
dat2=r'c:\Users\Arushi Mathur\Desktop\stockcompany.xlsx'


# In[14]:


df1.to_excel(dat2)


# In[15]:


df10=df1.loc[df1['Selected'] == 1]


# In[16]:


df10.head()


# In[17]:


datasource11=r'd:\Stock\finalstock.xlsx'


# In[18]:


df10.to_excel(datasource11)


# In[ ]:


# Implementing Logistic Regression ##
from sklearn.linear_model import LogisticRegression


# In[14]:


model = LogisticRegression()


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X = df1[['Median']]


# In[17]:


Y=df1[['Selected']]


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.4)


# In[19]:


X_test


# In[20]:


Y_test


# In[21]:


model.fit(X_train, Y_train)


# In[22]:


Y_predicted = model.predict(X_test)


# In[23]:


model.predict_proba(X_test)
#this means that , see 0.9878..(the first one) , it says that there is 98% probability that the stock will be selected.


# In[24]:


model.score(X_test,Y_test)


# In[25]:


model.score(X_test,Y_test)


# In[26]:


X_test


# In[27]:


Y_predicted


# In[ ]:


#Implementing Linear Regression 


# In[28]:


X=df1[['Median']]


# In[29]:


Y=df1[['Daychg']]


# In[30]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.4)


# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


clf = LinearRegression()


# In[33]:


clf.fit(X_test, Y_test)


# In[34]:



clf.score(X_test, Y_test)


# In[ ]:


## Done Linear Regression 


# In[51]:


import matplotlib.pyplot as plt


# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


plt.scatter(df1.Selected,df1.Median,marker='+',color='red')


# In[54]:


df1.shape


# In[55]:


model.predict(X_test)


# In[56]:


X_test


# In[57]:


Y_test


# In[58]:


## SVM Algorithm 
import re


# In[59]:


import operator


# In[60]:


import sys


# In[61]:


import os


# In[62]:


from sklearn.svm import SVR


# In[63]:


import matplotlib.pyplot as matplt


# In[64]:


svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)


# In[91]:


X = df1[['Median']]


# In[92]:


Y=df1[['Selected']]


# In[93]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.4)


# In[94]:


svr_lin.fit(X_train,Y_train)
svr_poly.fit(X_train,Y_train)
svr_rbf.fit(X_train,Y_train)


# In[95]:


matplt.scatter(X_train,Y_train,color='black',label='data')


# In[96]:



matplt.plot(X_train,svr_lin.predict(X_train),color='blue',label='Linear SVR')


# In[97]:


matplt.plot(X_train,svr_poly.predict(X_train),color='red',label='Polynomial SVR')


# In[98]:



matplt.plot(X_train,svr_rbf.predict(X_train),color='green',label='RBF SVR')


# In[ ]:




