#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


data = pd.read_csv('iris.csv')

data.head()
# In[46]:


data.columns


# In[47]:


data.describe()


# In[48]:



data.shape


# In[49]:


plt.figure(1)
plt.boxplot([data['PetalLengthCm']])
plt.figure(2)
plt.boxplot([data['SepalLengthCm']])
plt.show()


# In[50]:


sns.boxplot(data=data,x='Species',y='SepalWidthCm')


# In[51]:


sns.pairplot(data,hue='Species')


# In[52]:


sns.heatmap(data.corr(),annot=True)


# In[53]:


data.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)


# In[54]:


X= data['SepalLengthCm'].values.reshape(-1,1)
Y = data['SepalWidthCm'].values.reshape(-1,1)
plt.scatter(X,Y ,color='b')
plt.show()


# In[55]:


corr_mat = data.corr()
print(corr_mat)


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[57]:


train, test = train_test_split(data,test_size= 0.25)
print(train.shape)
print(test.shape)
data.columns


# In[58]:


train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
train_y = train.Species

test_X =  test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
test_y = test.Species


# In[59]:


train_y.head()


# In[60]:


train_X.head()


# In[61]:


model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[62]:


from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)


# In[63]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,test_X,test_y)
plt.show


# In[ ]:





# In[ ]:




