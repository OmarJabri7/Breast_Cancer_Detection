#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
import string
import csv


# In[14]:


dataset = pd.read_csv('data.csv')


# In[15]:


dataset


# In[16]:


dataset


# In[17]:


dataset.describe()


# In[18]:


X = dataset.iloc[:,1:18].values


# In[19]:


X


# In[20]:


Y = dataset.iloc[:,0:1].values


# In[21]:


Y


# In[22]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# sc_Y = StandardScaler()
# Y = sc_Y.fit_transform(Y.reshape(-1,1))


# In[24]:


X


# In[25]:


Y


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 49)


# In[27]:


X_train.shape


# In[28]:


X_test.shape


# In[29]:


Y_train.shape


# In[30]:


Y_test.shape


# In[31]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 49)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)


# In[32]:


Y_test


# In[33]:


Y_pred


# In[34]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[35]:


cm


# In[36]:


from sklearn.metrics import average_precision_score
score = average_precision_score(Y_test,Y_pred)
score


# In[37]:


from sklearn.metrics import balanced_accuracy_score
score = balanced_accuracy_score(Y_test,Y_pred)
score


# In[38]:


from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test,Y_pred)
score


# In[39]:


from sklearn.metrics import f1_score
score = f1_score(Y_test,Y_pred)
score


# In[40]:


from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(15)]).T
pred = classifier.predict(Xpred).reshape(X1.shape)
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[41]:


X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(15)]).T
pred = classifier.predict(Xpred).reshape(X1.shape)
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




