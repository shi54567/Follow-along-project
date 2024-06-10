#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import the libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


# display all the rows and columns
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[17]:


# Load the dataset
df=pd.read_csv("auto-mpg.csv")


# In[18]:


# Shape
df.shape


# In[19]:


# datatypes
df.dtypes


# In[20]:


# Missing values??
df.isnull().sum()


# In[21]:


df.head()


# In[22]:


df["hp"].value_counts()


# In[23]:


df["hp"]=df["hp"].replace("?",np.nan)
df["hp"]=df["hp"].astype(float)


# In[24]:


df.dtypes


# In[25]:


df.describe()


# In[26]:


df["hp"]=df["hp"].replace(np.nan,df["hp"].median())


# In[27]:


df.describe()


# In[28]:


# Change origin into catgoric->
df["origin"]=df["origin"].replace({1:"america",2:"europe",3:"asia"})


# In[29]:


df.sample(10)


# In[30]:


df.hist(figsize=(10,12))
plt.show()


# In[31]:


df.skew(numeric_only=True)


# In[32]:


sns.boxplot(x="mpg",data=df)


# In[33]:


sns.boxplot(x="cyl",data=df)


# In[34]:


import plotly.express as px
for column in df:
    fig=px.histogram(df,x=column,nbins=20)
    fig.show()


# In[35]:


for column in df:
    fig=px.box(df,x=column)
    fig.show()


# In[36]:


sns.pairplot(df)


# In[37]:


sns.countplot(x="origin",data=df)


# In[38]:


sns.barplot(x="cyl",y="mpg",data=df)


# In[39]:


sns.lineplot(x="yr",y="mpg",data=df)


# In[40]:


sns.jointplot(x="wt",y="mpg",data=df)


# In[41]:


corr_matrix=df.corr(numeric_only=True)
corr_matrix


# In[42]:


sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")


# In[43]:


sns.kdeplot(x="wt", y="mpg", data=df)


# In[44]:


sns.boxenplot(x="origin", y="mpg", data=df)


# In[ ]:


g = sns.FacetGrid(col='origin', row='yr',data=df)
g.map(sns.scatterplot, 'cyl', 'acc')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('FacetGrid Plot')
plt.show()


# In[ ]:


sns.pairplot(data=df, hue="mpg", diag_kind="kde", palette="husl")


# In[ ]:




