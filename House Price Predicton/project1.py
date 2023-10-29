#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


data = pd.read_csv(r"C:\bharat intern projects\house-price-prediction-master\house-price-prediction-master\kc_house_data.csv")


# In[43]:


data.head()


# In[44]:


data.describe()


# In[45]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[46]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
plt1 = plt()
sns.despine


# In[51]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")


# In[52]:


plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")


# In[54]:


train1 = data.drop(['id', 'price'],axis=1)


# In[53]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")


# In[66]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[67]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[55]:


train1.head()


# In[56]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)


# In[71]:


reg.fit(x_train,y_train)
prediction = reg.predict(x_train)


# In[72]:


reg.score(x_test,y_test)


# In[ ]:





# In[ ]:




