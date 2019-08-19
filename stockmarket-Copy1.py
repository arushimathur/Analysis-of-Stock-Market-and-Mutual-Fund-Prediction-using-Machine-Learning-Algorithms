#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy 
import csv
import pandas as pd


# In[2]:


consumer_key = 'd7bw4FdO2UKG42xqtpnLolJon' 
consumer_secret = 'cwIGwLkkJkvL2mOUiTKrB2ogA5KqvWaMvgRNvzpx2TnGJ6ZUsx'
access_token = '777394875964678145-7TlPRYvlVB9KKJJx3ZJFeMoWgxcLpXN'  
access_token_secret = 'm4GxM6D8FE7IXqj2Xf6PIS8RH0kjltPaCLCcezMxGtaFX'


# In[3]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret) 
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines # Open/Create a file to append data
csvFile = open('ua.csv', 'a') #Use csv Writer
csvWriter = csv.writer(csvFile)
for tweet in tweepy.Cursor(api.search,q="#mutual",count=100, 
                           lang="en", 
                           since="2018-02-16").items(): 
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


# In[ ]:





# In[ ]:





# In[ ]:




