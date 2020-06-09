#!/usr/bin/env python
# coding: utf-8

# <img src="https://rhyme.com/assets/img/logo-dark.png" align="center"> 
# 
# <h2 align="center">Multiple Linear Regression</h2>

# Linear Regression is a useful tool for predicting a quantitative response.

# We have an input vector $X^T = (X_1, X_2,...,X_p)$, and want to predict a real-valued output $Y$. The linear regression model has the form

# <h4 align="center"> $f(x) = \beta_0 + \sum_{j=1}^p X_j \beta_j$. </h4>

# The linear model either assumes that the regression function $E(Y|X)$ is linear, or that the linear model is a reasonable approximation.Here the $\beta_j$'s are unknown parameters or coefficients, and the variables $X_j$ can come from different sources. No matter the source of $X_j$, the model is linear in the parameters.

# **Simple Linear Regression**: <h5 align=center>$$Y = \beta_0 + \beta_1 X + \epsilon$$</h5>

# **Multiple Linear Regression**: <h5 align=center>$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 +...+ \beta_p X_p + \epsilon$$ </h5>
# <h5 align=center> $$sales = \beta_0 + \beta_1 \times TV + \beta_2 \times radio + \beta_3 \times newspaper + \epsilon$$ </h5>

# ### Task 1: Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)


#  

# ### Task 2: Load the Data

# The adverstiting dataset captures sales revenue generated with respect to advertisement spends across multiple channles like radio, tv and newspaper.

# In[5]:


advert =  pd.read_csv("Advertising.csv")
advert.head()


# In[6]:


advert.info()


# ### Task 3: Relationship between Features and Response

# In[8]:


sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, 
            aspect=0.7)


#  

# ### Task 4: Multiple Linear Regression - Estimating Coefficients

# In[9]:


from sklearn.linear_model import LinearRegression
X = advert[['TV', 'radio', 'newspaper']]
y = advert.sales

lm1 = LinearRegression()
lm1.fit(X,y)

print(lm1.intercept_)
print(lm1.coef_)


# In[11]:


list(zip(['TV', 'radio', 'newspaper'], lm1.coef_))


# In[13]:


sns.heatmap(advert.corr(), annot=True)


#  

#  

#  

# ### Task 5: Feature Selection

#  

# In[20]:


from sklearn.metrics import r2_score
lm2 = LinearRegression().fit(X[['TV', 'radio']] , y)
lm2_pred = lm2.predict(X[['TV', 'radio']])

print("R^2", r2_score(y,lm2_pred))


# In[21]:


lm3 = LinearRegression().fit(X[['TV', 'radio', 'newspaper']] , y)
lm3_pred = lm3.predict(X[['TV', 'radio','newspaper']])

print("R^2", r2_score(y,lm3_pred))


#  

# ### Task 6: Model Evaluation Using Train/Test Split and Metrics

#  

# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors: <h5 align=center>$$\frac{1}{n}\sum_{i=1}^{n} \left |y_i - \hat{y_i} \right |$$</h5>
# **Mean Squared Error** (MSE) is the mean of the squared errors: <h5 align=center>$$\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y_i})^2$$</h5>
# **Root Mean Squared Error** (RMSE) is the mean of the squared errors: <h5 align=center>$$\sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y_i})^2}$$</h5>

#  

#  

#  

# Let's use train/test split with RMSE to see whether newspaper should be kept in the model:

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X =  advert[['TV', 'radio', 'newspaper']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

lm4 = LinearRegression().fit(X_train, y_train)
lm4_pred = lm4.predict(X_test)

print("RSME:", np.sqrt(mean_squared_error(y_test, lm4_pred)))
print("R^2:", r2_score(y_test, lm4_pred))


# In[27]:



X =  advert[['TV', 'radio']]
y = advert.sales

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

lm5 = LinearRegression().fit(X_train, y_train)
lm5_pred = lm5.predict(X_test)

print("RSME:", np.sqrt(mean_squared_error(y_test, lm5_pred)))
print("R^2:", r2_score(y_test, lm5_pred))


# In[29]:


from yellowbrick.regressor import PredictionError, ResidualsPlot

visualizer = PredictionError(lm5).fit(X_train,y_train)
visualizer.score(X_test, y_test)
visualizer.poof()

#multiple regression line


# In[ ]:





#  

# ### Task 7: Interaction Effect (Synergy)

# In[30]:


advert['interaction'] = advert['TV'] * advert['radio']

X= advert[['TV', 'radio', 'interaction']]
y= advert.sales

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

lm6 = LinearRegression().fit(X_train, y_train)
lm6_pred = lm6.predict(X_test)

print("RSME:", np.sqrt(mean_squared_error(y_test, lm6_pred)))
print("R^2:", r2_score(y_test, lm6_pred))


# In[31]:


visualizer = PredictionError(lm6).fit(X_train,y_train)
visualizer.score(X_test, y_test)
visualizer.poof()

#multiple regression line


# In[ ]:




