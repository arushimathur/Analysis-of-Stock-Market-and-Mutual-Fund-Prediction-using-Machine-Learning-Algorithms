#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader.data as web


# In[2]:


import datetime as dr


# In[3]:


start = dr.datetime(2015,2,15)


# In[4]:


end=dr.datetime.today()


# In[7]:


stock='AAPL'


# In[8]:


df=web.DataReader(stock,'yahoo',start,end)


# In[9]:


print(df.head())


# In[10]:


df=df.rename(columns={'Adj Close':'Close'})


# In[11]:


print(df.head())


# In[2]:


data_source=r'd:\Stock\AAPL.xlsx'


# In[3]:


import pandas as pd


# In[18]:


df.to_excel(data_source)


# In[4]:


import numpy as np


# In[5]:


data_source=r'd:\Stock\AAPL.xlsx'


# In[6]:


df1=pd.read_excel(data_source,index_col='Date')


# In[7]:


ndayforward=2


# In[8]:


df1['day_chg']=(df1['Close'].pct_change())*100


# In[9]:


print(df1.tail())


# In[25]:


df1['ndaychg']=df1['day_chg'].shift(-ndayforward)


# In[26]:


docriteria=(df1.index>='2018-01-1') & (df1.index<=dr.datetime.today())


# In[20]:


tcriteria=df1['day_chg']<-1


# In[21]:


criteria=(docriteria) & (tcriteria)


# In[22]:


print(df1[criteria].tail())


# In[23]:


df1.to_excel(data_source)


# In[24]:


print(df1[criteria].loc[:,['Close','ndaychg']].agg(['mean','median','std']))


# In[25]:


print(df1[criteria].loc[:,['Close','day_chg']].agg(['mean','median','std']))


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


df1.info()


# In[12]:


X = df1[['Date1']]


# In[13]:


Y=df1[['Close.1']]


# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.2)


# In[15]:


from sklearn.svm import SVR


# In[16]:


import matplotlib.pyplot as matplt


# In[17]:


svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)


# In[ ]:


svr_lin.fit(X_train,Y_train)


# In[ ]:


svr_poly.fit(X_train,Y_train)


# In[ ]:


svr_rbf.fit(X_train,Y_train)


# In[ ]:


matplt.scatter(X_train,Y_train,color='black',label='data')


# In[37]:



matplt.plot(X_train,svr_lin.predict(X_train),color='blue',label='Linear SVR')


# In[38]:


matplt.plot(X_train,svr_poly.predict(X_train),color='red',label='Polynomial SVR')


# In[39]:


matplt.plot(X_train,svr_rbf.predict(X_train),color='green',label='RBF SVR')


# In[42]:


matplt.xlabel('X_train')


# In[43]:


matplt.ylabel('Y_train')


# In[44]:


matplt.title('Support Vector Regression')


# In[45]:


matplt.legend()


# In[46]:


matplt.show()
#########


# In[27]:


df2=pd.read_excel(data_source,index_col='Date')


# In[ ]:





# In[28]:


data_source2=r'd:\Stock\DOW jonesagg.xlsx'


# In[29]:


df3=df1[criteria].loc[:,['Close','day_chg']].agg(['mean','median','std'])


# In[30]:


df3.to_excel(data_source2)


# In[31]:


stock2='UNH'


# In[32]:


df4=web.DataReader(stock2,'yahoo',start,end)


# In[33]:


print(df4.head())


# In[34]:


df4=df4.rename(columns={'Adj Close':'Close'})


# In[35]:


data_source3=r'd:\Stock\UnitedHealthcare.xlsx'


# In[36]:


df4.to_excel(data_source3)


# In[37]:


df5=pd.read_excel(data_source3,index_col="Date")


# In[38]:


ndayforward=2


# In[39]:


df5['day_chg']=(df5['Close'].pct_change())*100


# In[40]:


print(df5.tail())


# In[41]:


df5['nadaychg']=df5['day_chg'].shift(-ndayforward)


# In[42]:


docriteria1=(df5.index>='2018-01-1') & (df4.index<=dr.datetime.today())


# In[43]:


tcriteria1=df5['day_chg']<-1


# In[44]:


criteria1=(docriteria1) & (tcriteria1)


# In[45]:


print(df5[criteria1].tail())


# In[46]:


df5.to_excel(data_source3)


# In[47]:


print(df5[criteria1].loc[:,['Close','nadaychg']].agg(['mean','median','std']))


# In[48]:


print(df5[criteria1].loc[:,['Close','day_chg']].agg(['mean','median','std']))


# In[49]:


data_source4=r'd:\Stock\UnitedHealthcareagg.xlsx'


# In[50]:


df7=df5[criteria1].loc[:,['Close','day_chg']].agg(['mean','median','std'])


# In[51]:


df7.to_excel(data_source4)


# In[52]:


import matplotlib.pyplot as plt


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


plt.scatter(df5['day_chg'],df5['Close'])


# In[55]:


plt.scatter(df5['nadaychg'],df5['Close'])


# In[56]:


X = df5[['Close']]


# In[57]:


Y=df5[['day_chg']]


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)


# In[60]:


X_train


# In[61]:




X_test


# In[62]:


Y_train


# In[63]:


Y_test


# In[64]:


from sklearn.linear_model import LinearRegression


# In[65]:


clf = LinearRegression()


# In[66]:


clf.fit(X_test, Y_test)


# In[67]:


X_test


# In[68]:


clf.predict(X_test)


# In[69]:


Y_test


# In[70]:




clf.score(X_test, Y_test)


# In[71]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
X_test

