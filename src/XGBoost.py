"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import pandas as pd
import os
os.chdir("F:\\03-github\\turnusol\\turnusol\\experiments")
data = pd.read_csv("dataset/train.csv").dropna()
cv = CountVectorizer(max_features=5000, encoding="utf-8",  
      ngram_range = (1,3),  
      token_pattern = "[A-Za-z_][A-Za-z\d_]*")
X = cv.fit_transform(data.title).toarray()
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
      test_size=0.33, 
      random_state=0)
count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())

count_df['etiket'] = y_train
data.sample(5)
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import warnings  
warnings.filterwarnings('ignore')

# importing packages
import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# sklearn packages
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# nltk packages
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from string import punctuation
import unidecode


data = pd.read_csv('dataset/train.csv')
print(data.head(5))
