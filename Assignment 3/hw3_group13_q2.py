# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

import math

# %% [markdown]
# For us.txt:

# %%
us = pd.read_csv("D:/IUB/COURSE WORK/FALL 2022/APPLIED MACHINE LEARNING/ASSIGNMENTS/ASSIGNMENT 3/us.txt", header = None)

# %%
us.head()

# %% [markdown]
# adding a column "country":

# %%
us["country"] = "us"

# %% [markdown]
# adding header to the dataframe:

# %%
us.columns = ["name", "country"]

# %%
us.head()

# %% [markdown]
# For greek.txt:

# %%
greek = pd.read_csv("D:\IUB\COURSE WORK\FALL 2022\APPLIED MACHINE LEARNING\ASSIGNMENTS\ASSIGNMENT 3\greek.txt", header = None)

# %%
greek.head()

# %% [markdown]
# adding a column "country":

# %%
greek["country"] = "greek"

# %% [markdown]
# adding header to the dataframe:

# %%
greek.columns = ["name", "country"]

# %%
greek.head()

# %% [markdown]
# For japan.txt:

# %%
japan = pd.read_csv("D:\IUB\COURSE WORK\FALL 2022\APPLIED MACHINE LEARNING\ASSIGNMENTS\ASSIGNMENT 3\japan.txt", header = None)

# %%
japan.head()

# %% [markdown]
# adding a column "country":

# %%
japan["country"] = "japan"

# %% [markdown]
# adding header to the dataframe:

# %%
japan.columns = ["name", "country"]

# %%
japan.head()

# %% [markdown]
# *For* arabic.txt:

# %%
arabic = pd.read_csv("D:/IUB/COURSE WORK/FALL 2022/APPLIED MACHINE LEARNING/ASSIGNMENTS/ASSIGNMENT 3/arabic.txt", header = None)

# %%
arabic.head()

# %% [markdown]
# adding a column "country":

# %%
arabic["country"] = "arabic"

# %% [markdown]
# adding header to the dataframe:

# %%
arabic.columns = ["name", "country"]

# %%
arabic.head()

# %% [markdown]
# **combining all the dataframes into single dataframe:**
# 
# ---
# 
# 

# %%
df = pd.concat([us, greek, japan, arabic])

# %%
df.head()

# %%
df.shape

# %%
df.info()

# %% [markdown]
# **Preprocessing**:
# 
# Using countVectorizer for column 'name'

# %%
vectorizer = CountVectorizer().fit(df['name'])

# %%
vec_names = vectorizer.transform(df['name'])

# %% [markdown]
# Splitting the dataframe into trining and testing datasets:

# %%
(X_train, X_test, y_train, y_test) = train_test_split(vec_names, df["country"], shuffle = True)

# %%
#using function from sklearn
from sklearn.naive_bayes import MultinomialNB
# # instantiate the model as clf(classifier) and train it
clf = MultinomialNB()
clf.fit(X_train, y_train)

# %%
pred_test_gaussian = clf.predict(X_test)
print(accuracy_score(y_test, pred_test_gaussian) * 100)

# %%
tname = ["john"]
vt = vectorizer.transform(tname)
predt = clf.predict(vt)
print(predt)

# %% [markdown]
# ####**Naive Bayes:**

# %%
#function for mean calculation
def mean(n):
  sum = sum(n)
  mean = sum / float(len(n))
  return mean

# %%
#function for standard deviation clculation
def s_dev(n):
    mean = mean(n)
    sq_mean = [pow(x - mean, 2) for x in n]
    var = sum(sq_mean) / float(len(n) - 1)
    sd = math.sqrt(var)
    return sd

# %%
#function for P(Y)
def p_y(x, y):
    y = np.concatenate([y, [-1,1]])
    n = len(y)
    pos = len(y[y == 1])/n
    return pos

pos = p_y(X_train,y_train)

# %%


# %% [markdown]
# ###All References
# 
# Q.1 
# 1. https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118 
# 2. https://medium.com/@pranjallk1995/pca-for-image-reconstruction-from-scratch-cf4a787c1e36 
# 3. https://vitalflux.com/pca-explained-variance-concept-python-example/#:~:text=Explained%20variance%20is%20calculated%20as,decomposition%20PCA%20class
# 
# 
# Q.2
# 
# 1. https://towardsdatascience.com/name-classification-with-naive-bayes-7c5e1415788a
# 2. https://github.com/joepatten/machine_learning_practice/blob/master/naive_bayes/naive_bayes.ipynb 


