#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A=pd.read_csv("C:/Users/Neha/Downloads/PythonNotes/Cars93.csv")


# Created dataframe from csv file cars93

# In[2]:


A.head()


# Preview of the dataset

# # Univariate Analysis
# 
Analyse the relationship just under one column.
For continous columns we plot Distribution plot or Histogram.
For categorical columns we have countplot,bar graph,pie chart.
# In[3]:


#We can check categorical and continous columns
cat=[]
con=[]
for i in A.columns:
    if(A[i].dtype=='object'):
        cat.append(i)
    else:
        con.append(i)


# In[4]:


cat


# In[5]:


con


# In[6]:


#Another way to know categorical and continous columns inside the dataset
A.info()


# In[8]:


import seaborn as sb
sb.distplot(A.Price)


# In[4]:


sb.distplot(A['MPG.city'])


# In[10]:


#Histogram
A['Price'].hist()


# In[11]:


A['MPG.city'].hist()


# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sb.distplot(A['Price'])
plt.subplot(2,2,2)
A['Price'].hist()
plt.xlabel("Price")
plt.subplot(2,2,3)
sb.distplot(A['MPG.city'])
plt.subplot(2,2,4)
A['MPG.city'].hist()
plt.xlabel("MPG.city")

Distribution plot on y-axis gives the proportion of data within given interval and ditribution plot has distribution curve which represents the spread of data around the mean
Histogram plot on y-axis gives the number of observation within given interval.
# In[18]:


A.Type.value_counts()


# In[19]:


A.Type.value_counts().plot(kind='barh')


# In[20]:


A.Type.value_counts().plot(kind='bar')


# In[21]:


sb.countplot(A.Type)


# In[22]:


A.AirBags.value_counts().plot(kind='pie')


# # Bivariate Analysis
Analyse the relationship between the two columns
con,con->scatterplot,correlation coefficient
con,cat->Anova,Boxplot
cat,cat->Crosstab,countplot with hue
# In[16]:


import numpy as np
np.corrcoef(A.Price,A['MPG.city'])


# In[23]:


A.corr()


# In[24]:


A.corr()['Price']


# In[25]:


plt.scatter(A.Price,A.Horsepower)


# In[34]:


def Anova(CON,CAT,Df):
    from statsmodels.api import OLS
    from statsmodels.formula.api import ols
    rel=CON+"~"+CAT
    model=ols(rel,Df).fit()
    from statsmodels.stats.anova import anova_lm
    anova_result=anova_lm(model)
    Q=pd.DataFrame(anova_result)
    a=Q['PR(>F)'][CAT]
    print("%.40f"%a)


# In[37]:


Anova("Price","Type",A)


# In[39]:


Anova("Weight","Type",A)


# We can say that Price and weight columns have good relation with Type independantly as p value is less than 0.05

# In[41]:


sb.boxplot(A.Type,A.Price)


# In[45]:


sb.boxplot(A.Type,A.Price,order=['Small','Sporty','Compact','Van','Large','Midsize'])


# In[46]:


sb.boxplot(A.Type,A.Weight)


# In[48]:


sb.boxplot(A.Type,A.Weight,order=['Small','Compact','Sporty','Midsize','Large','Van'])


# Boxplot tells us by having change in the value of one continous column is having change in the categorical column or not.

# In[49]:


pd.crosstab(A.Type,A.AirBags)


# In[50]:


pd.crosstab(A.Type,A.Origin)


# cross tabulation is the numeric way to get the relationship between two categorical columns.Here it is showing the proportion of Type column inside Origin column.

# In[51]:


sb.countplot(A.Type,hue=A.Origin)


# In[52]:


sb.countplot(A.Type,hue=A.AirBags)


# Countplot gives count of each category inside the categorical column and hue determines which column should be used for color 
# encoding

# # Multivariate analysis
# Analyse the relationship for more than two columns: scatterplot with hue,heat map

# In[10]:


sb.scatterplot(A.Weight,A.Price,hue=A.Type)


# In[12]:


sb.heatmap(A.corr())


# Heatmap is used to understand correlations of all continous columns when dataset has large number of continous columns.

# In[ ]:




