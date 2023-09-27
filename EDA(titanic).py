#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# ##### EDA is an important step in ML project . This is we get to dig into data , trying to understand more about it.
# ##### By performing EDA , we can :
# ##### 1) Know  the summary statistics
# ##### 2) spot if there is any missing values and appropriate strategy for handling it
# ##### 3) spot if the data is skewed( or imbalanced)
# #### 4) correlation between features
# #### 5) understand the important features and unhelpful features
# #### 6) Above all answer some of the pressing questions about the data . these questions can be specific to data and its features but will around things like why this , and this or what could have caused this and this based on the analysis etc .... There are no right or wrong questions , the idea here is to use to see if we can help answer some questions  

# # 1. Import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### 2. Load Dataset

# In[3]:


titanic = sns.load_dataset('titanic')


# ### 3. Quick Look in dataset

# In[4]:


titanic.info()


# In[5]:


titanic.head()


# In[6]:


titanic.tail()


# # 4. Summary Statistics

# In[7]:


titanic.describe()


# In[8]:


titanic.describe().transpose()


# ## 5. Basic Information

# ###### 1. How many people who survived ad died from titanic crash ? Can you use appropriate visualization to show these people ?

# In[9]:


titanic['survived'].value_counts()


# In[10]:


sns.countplot(data = titanic , x = 'survived')


# ###### 1. how many pclass are there in dataset ? it is same as text column class ?

# In[12]:


titanic['pclass'].value_counts()


# In[13]:


titanic['class'].value_counts()


# In[14]:


p_class = titanic[['pclass','class']]
p_class.head()


# #### It seems that both of these columns are same except that one is numeric and other is text . They basically contain the same information of the class that passengers paid for .

# In[15]:


titanic['sex'].value_counts()


# In[16]:


sns.countplot(data = titanic , x = 'sex')


# In[17]:


less_than_20 = titanic[titanic['age'] < 20 ]

less_than_20


# In[18]:


len(less_than_20)


# In[19]:


titanic.who.value_counts().plot(kind = 'pie')


# #### How many unique cities are there in column embark_town ? Plot their occurences.

# In[24]:


titanic['embark_town'].nunique()


# In[25]:


titanic['embark_town'].value_counts().plot(kind = 'bar')


# ###### there is no limit to how deep you can go to understand the dataset.

# ## 6. Missing Data

# In[27]:


titanic.isnull().sum()


# ##### There are missing data in age and deck columns . We can also use heatmaps to show the missing values

# In[30]:


sns.heatmap(titanic.isnull() , yticklabels = False , cbar = False)


# ### 7. More Analysis

# In[33]:


sns.countplot(data = titanic , x = 'survived' , palette = 'autumn' , hue = 'sex')


# In[34]:


sns.countplot(data = titanic , x = 'survived' , palette = 'viridis' , hue = 'class')


# In[35]:


titanic.age.plot(kind = 'hist' , bins = 30 , color = 'green')


# In[36]:


sns.scatterplot(data = titanic , x = 'age' , y = 'fare' , hue = 'class')


# ###### this is obvious . the passengers who were in first class paid more than other classes . And many females were in first class

# In[37]:


sns.scatterplot(data = titanic , x = 'age' , y = 'fare' , hue = 'sex')


# ###### The purpose of this lab was to learn about exploratory analysis . There is no limit to what you can do . The more time you spend with the data , the good your analysis will be. I used questions to make it simple and lead way but there is no proper format for this . The goal is to be one with data

# ## 8. Checking Correlating Features

# ###### Checking correlation can help to see the similiarity between features . If two features correlate , that means they contain same information and if one of them is removed , the analysis/model can be less affected

# In[38]:


correlation = titanic.corr()


# In[39]:


correlation


# In[41]:


correlation['survived']


# ###### If you want to check features correlate with the labels here how to do it

# In[42]:


### Visualizing correlation

plt.figure(figsize = (12,7))

sns.heatmap(correlation , annot = True , cmap = 'crest')


# In[ ]:




