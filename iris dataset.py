#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split


# In[2]:


cr=pd.read_csv("C:\\Users\\Thinkpad\\Desktop\Iris.csv")


# In[3]:


cr.head()


# In[23]:


cr.info()


# In[24]:


x=cr.iloc[:,:4]
x.head()


# In[25]:


y=cr.iloc[:,-1]
y.head()


# In[26]:


# data normalization


# In[27]:


x = preprocessing.StandardScaler().fit_transform(x)
x[0:4]


# In[28]:


## train-test data


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size = 0.20,random_state=1)
y_test.shape


# In[39]:


# training and testing predicting


# In[45]:


knnmodel=KNeighborsClassifier(n_neighbors=5)
knnmodel.fit(x_train,y_train)
y_predict1=knnmodel.predict(x_test)


# In[41]:


# accuracy


# In[43]:


from sklearn.metrics  import accuracy_score


# In[46]:


acc=accuracy_score(y_test,y_predict1)
acc


# In[47]:


# confusion matrix


# In[49]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values,y_predict1)
cm


# In[50]:


cm1=pd.DataFrame(data=cm,index=['setosa','versicolor','verginica'],columns=['setosa','versicolor','virginica'])
cm1


# In[51]:


# prediction output


# In[52]:


prediction_output=pd.DataFrame(data=[y_test.values,y_predict1],index=['y_test','y_predict1'])


# In[53]:


prediction_output.transpose()


# In[ ]:




