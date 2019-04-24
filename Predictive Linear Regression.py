#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read data
df = pd.read_csv("Fuel_Consumption.csv")


# In[3]:


df.head()


# In[4]:


#summarize data
df.describe()


# In[5]:


features1 = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]


# In[6]:


features1.head()


# In[15]:


viz = features1[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]


# In[16]:


#plot each feature
viz.hist()
plt.show()


# In[18]:


plt.scatter(features1.FUELCONSUMPTION_COMB, features1.CO2EMISSIONS, color = 'blue')
plt.xlabel("Fuel_Consumption_Comb")
plt.ylabel("Emission")
plt.show()


# In[20]:


#show fuel consumption vs cylinders
plt.scatter(features1.FUELCONSUMPTION_COMB, features1.CYLINDERS, color = 'green')
plt.xlabel('Fuel_Consumption_Comb')
plt.ylabel('Cylinders')
plt.show()


# In[24]:


#cylinders vs emission
plt.scatter(features1.CYLINDERS, features1.CO2EMISSIONS, color = "Red")
plt.xlabel('Cylinders')
plt.ylabel('Emissions')
plt.title("Cylinders vs Emissions")
plt.show()


# In[26]:


#split the data set
mtr  = np.random.rand(len(df)) < 0.8
train_set = features1[mtr]
test_set = features1[mtr]


# In[27]:


#import sk learn
from sklearn import linear_model


# In[39]:


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train_set[['ENGINESIZE']])
train_y = np.asanyarray(train_set[['CO2EMISSIONS']])


# In[40]:


#fit arrays
regr.fit(train_x, train_y)


# In[43]:


#print coefficients
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)


# In[44]:


#plot line to fit data to model
plt.scatter(train_set.ENGINESIZE, train_set.CO2EMISSIONS, color = 'blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("Co2")
plt.title("Co2 vs Engine Size")


# In[52]:


#evaluate model
from sklearn.metrics import r2_score


# In[53]:


test_x = np.asanyarray(test_set[['ENGINESIZE']])
test_y = np.asanyarray(test_set[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)


# In[58]:


#Print Stats
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2: %.2f" % r2_score(test_y_hat , test_y) )


# In[ ]:




