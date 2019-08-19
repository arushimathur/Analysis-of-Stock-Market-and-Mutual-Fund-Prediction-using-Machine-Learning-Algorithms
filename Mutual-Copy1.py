#!/usr/bin/env python
# coding: utf-8

# In[3]:


import bs4


# In[4]:


import requests


# In[21]:


res=requests.get('http://www.moneycontrol.com/mf/mfinfo/amc_sch_listing.php?ffid=BS')


# In[22]:


type(res)


# In[23]:


soup=bs4.BeautifulSoup(res.text,'lxml')


# In[24]:


soup.select('.boxBg1')


# In[25]:


for i in soup.select('.boxBg1'):
    print(i.text)


# In[1]:


import pandas as pd


# In[2]:


tables = pd.read_html("http://www.moneycontrol.com/mf/mfinfo/amc_sch_listing.php?ffid=BS")


# In[143]:


tables[0]


# In[35]:


datasource2=r'd:\Mutual\MutualFund1.xlsx'


# In[3]:


df=pd.read_excel(datasource2)


# In[4]:


print(df.head())


# In[123]:




df.fillna(0)


# In[5]:


datasource7=r'd:\Mutual\MutualFund2.xlsx'


# In[125]:


df.to_excel(datasource6)


# In[6]:


df3=pd.read_excel(datasource7)


# In[7]:


print(df3.head())


# In[8]:


import numpy as np


# In[9]:


print(df3.loc[:,['LatestNAV']].agg(['mean','median','std']))


# In[10]:


df4=df3.copy()


# In[11]:


df4['Selected']=np.where(df4['LatestNAV']>10.81,1,0)


# In[12]:


df4.head()


# In[13]:


datasource8=r'd:\Mutual\MutualFund3.xlsx'


# In[49]:


df4.to_excel(datasource8)


# In[ ]:





# In[14]:




import matplotlib.pyplot as plt



# In[15]:



get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X=df4[['LatestNAV']]


# In[19]:


Y=df4[['Selected']]


# In[20]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.3)


# In[21]:


X_test


# In[22]:


Y_test


# In[23]:


## Logistic Regression ##
from sklearn.linear_model import LogisticRegression


# In[24]:


model = LogisticRegression()


# In[25]:


model.fit(X_train,Y_train)


# In[26]:


Y_predicted = model.predict(X_test)


# In[27]:


model.predict_proba(X_test)
#this means that , see 0.9878..(the first one) , it says that there is 98% probability that the stock will be selected.


# In[28]:


model.score(X_test,Y_test)


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


plt.scatter(df4.LatestNAV,df4.Selected,marker='+',color='red')


# In[ ]:


## DONE WITH LOGISTIC REGRESSION ##


# In[73]:


## Let's Start with Decision Tree now 
df5=df4.copy()


# In[74]:


df5.head()


# In[75]:


inputs = df5.drop('Selected',axis='columns')


# In[76]:


target = df5['Selected']


# In[78]:


from sklearn.preprocessing import LabelEncoder
le_scheme = LabelEncoder()
le_nav = LabelEncoder()
le_assets = LabelEncoder()


# In[81]:


inputs['Scheme_n'] = le_company.fit_transform(inputs['Scheme Name'])
inputs['Nav_n'] = le_company.fit_transform(inputs['LatestNAV'])
inputs['Assets_n'] = le_company.fit_transform(inputs['Assets(Rs. cr.)'])


# In[82]:




inputs


# In[83]:


inputs_n = inputs.drop(['Scheme Name','Category','LatestNAV','1 yr Returns (%)','Assets(Rs. cr.)'],axis='columns')


# In[84]:


inputs_n


# In[85]:


target


# In[86]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[87]:


model.fit(inputs_n, target)


# In[88]:


model.score(inputs_n,target)


# In[89]:


model.predict([[12,152,346]])


# In[90]:


datasource8=r'd:\Mutual\MutualFund4.xlsx'


# In[92]:


inputs_n.to_excel(datasource8)


# In[ ]:


## Done with decision Tree Now ##


# In[10]:


df4.head()


# In[11]:


df6=df4.loc[df4['Selected'] == 1]


# In[12]:


df6.head()


# In[13]:


datasource9=r'd:\Mutual\MutualFund5.xlsx'


# In[14]:


df6.to_excel(datasource9)


# In[15]:


df7=df6.sort_values('LatestNAV',ascending=False)


# In[17]:


df8=df7.head(5)


# In[18]:


datasource=r'd:\Mutual\MutualFund6.xlsx'


# In[19]:


df8.to_excel(datasource)


# In[45]:


df12=df4.head(50)


# In[47]:


## Linear Regression Model ##
X = df12[['Assets(Rs. cr.)']]


# In[48]:


Y=df12[['LatestNAV']]


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)


# In[51]:


X_train


# In[52]:


Y_train


# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


clf = LinearRegression()


# In[55]:


clf.fit(X_test, Y_test)


# In[56]:


clf.predict(X_test)


# In[57]:


clf.score(X_test, Y_test)


# In[ ]:


## Done with Linear Regression ##


# In[30]:


## SVM Model ##
from sklearn.model_selection import train_test_split


# In[31]:


df3.info()


# In[32]:


from sklearn.svm import SVR


# In[33]:


import matplotlib.pyplot as matplt


# In[34]:


df9=df7.head(20)


# In[35]:


X=df9[['Assets(Rs. cr.)']]


# In[36]:


Y=df9[['LatestNAV']]


# In[40]:


Y


# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4)


# In[42]:


X_train


# In[43]:


Y_train


# In[44]:


svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)


# In[ ]:


svr_lin.fit(X_train,Y_train)
svr_poly.fit(X_train,Y_train)
svr_rbf.fit(X_train,Y_train)


# In[ ]:


matplt.scatter(X_train,Y_train,color='black',label='data')


# In[ ]:



matplt.plot(X_train,svr_lin.predict(X_train),color='blue',label='Linear SVR')

