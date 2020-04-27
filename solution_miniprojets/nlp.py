
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('logstop', '')
get_ipython().run_line_magic('logstart', '-ortq ~/.logs/nlp.py append')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


# In[4]:


from static_grader import grader


# # NLP Miniproject

# ## Introduction
# 
# The objective of this miniproject is to gain experience with natural language processing and how to use text data to train a machine learning model to make predictions. For the miniproject, we will be working with product review text from Amazon. The reviews are for only products in the "Electronics" category. The objective is to train a model to predict the rating, ranging from 1 to 5 stars.
# 
# ## Scoring
# 
# For most of the questions, you will be asked to submit the `predict` method of your trained model to the grader. The grader will use the passed `predict` method to evaluate how your model performs on a test set with respect to a reference model. The grader uses the [R<sup>2</sup>-score](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score) for model evaluation. If your model performs better than the reference solution, then you can score higher than 1.0. For the last question, you will submit the result of an analysis and your passed answer will be compared directly to the reference solution.
# 
# ## Downloading and loading the data
# 
# The data set is available on Amazon S3 and comes as a compressed file where each line is a JSON object. To load the data set, we will need to use the `gzip` library to open the file and decode each JSON into a Python dictionary. In the end, we have a list of dictionaries, where each dictionary represents an observation.

# In[5]:


get_ipython().run_cell_magic('bash', '', 'mkdir data\nwget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data')


# In[6]:


import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]


# The ratings are stored in the keyword `"overall"`. You should create an array of the ratings for each review, preferably using list comprehensions.

# In[7]:


ratings = [x['overall'] for x in data]


# In[8]:


data


# **Note**, the test set used by the grader is in the same format as that of `data`, a list of dictionaries. Your trained model needs to accept data in the same format. Thus, you should use `Pipeline` when constructing your model so that all necessary transformation needed are encapsulated into a single estimator object.

# ## Question 1: Bag of words model
# 
# Construct a machine learning model trained on word counts using the bag of words algorithm. Remember, the bag of words is implemented with `CountVectorizer`. Some things you should consider:
# 
# * The reference solution uses a linear model and you should as well; use either `Ridge` or `SGDRegressor`.
# * The text review is stored in the key `"reviewText"`. You will need to construct a custom transformer to extract out the value of this key. It will be the first step in your pipeline.
# * Consider what hyperparameters you will need to tune for your model.
# * Subsampling the training data will boost training times, which will be helpful when determining the best hyperparameters to use. Note, your final model will perform best if it is trained on the full data set.
# * Including stop words may help with performance.

# In[9]:


from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    
    def fit(self, X, y=None):
        return self;
    
    def transform(self, X):
        return [x[self.key] for x in X]  


# In[10]:


ks = KeySelector('reviewText')
X_trans = ks.fit_transform(data)


# In[11]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split


# In[12]:


X_mbatch1, X_mbatch2, y_mbatch1, y_mbatch2=X_train, X_test, y_train, y_test=train_test_split(data, ratings ,test_size=0.5, random_state=42)


# In[13]:


len(X_test)


# In[14]:



bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
#     ('vectorizer', CountVectorizer()), OR
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=1))
])

# bag_of_words_model.fit(data, ratings);
bag_of_words_model.fit(X_train, y_train);

# bag_of_words_trans = Pipeline([
#     ('selector', KeySelector('reviewText')),
#      ('vectorizer', HashingVectorizer()),
# ])


# In[15]:


# Mini-Batch 1
# X_trans=bag_of_words_trans.fit_transform(X_mbatch1)
# sgd = SGDRegressor(warm_start=True, random_state=42)
# # sgd.partial_fit(X_trans, y_mbatch1)
# sgd.fit(X_trans, y_mbatch1)
# print(sgd.coef_)

# Mini-Batch 2
# X_trans=bag_of_words_trans.fit_transform(X_mbatch2)
# sgd = SGDRegressor(warm_start=True, random_state=42)
# # sgd.partial_fit(X_trans, y_mbatch2)
# sgd.fit(X_trans, y_mbatch2)
# print(sgd.coef_)


# In[16]:


# # One Batch 
# sgd2 = SGDRegressor(random_state=42)
# X_trans = bag_of_words_trans.fit_transform(data)
# sgd2.fit(X_trans, ratings)
# print('One batch Coef:,  ', sgd2.coef_)


# In[17]:


# print('Online learning (min-batches): ', sgd.score(X_trans, ratings))


# In[18]:


# print('Batch learning : ', sgd2.score(X_trans, ratings))


# In[19]:


grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)


# ## Question 2: Normalized model
# 
# Using raw counts will not be as effective compared if we had normalized the counts. There are several ways to normalize raw counts; the `HashingVectorizer` class has the keyword `norm` and there is also the `TfidfTransformer` and `TfidfVectorizer` that perform tf-idf weighting on the counts. Apply normalized to your model to improve performance.

# In[20]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import numpy as np


# In[21]:


normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(ngram_range=(1,2))),
#      ('vectorizer', TfidfVectorizer()),
#     ('reduce', TruncatedSVD(n_components=100)),
    ('predictor', Ridge(alpha=0.6))
])


# In[22]:


params={'vectorizer__lowercase': [True, False],
       'vectorizer__stop_words': [None, 'english'],
       'vectorizer__norm':  ['l1','l2']}

gs = GridSearchCV(normalized_model, param_grid=params, cv=3, verbose=1, n_jobs=-1)


# In[23]:


normalized_model.fit(data, ratings);


# In[24]:


#normalized_model.named_steps['predictor'].best_params_


# In[25]:


grader.score.nlp__normalized_model(normalized_model.predict)


# In[26]:



bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('predictor', Ridge(alpha=0.6))
])
bigrams_model.fit(data, ratings);


# In[27]:


grader.score.nlp__bigrams_model(bigrams_model.predict)


# ## Question 4: Polarity analysis
# 
# Let's derive some insight from our analysis. We want to determine the most polarizing words in the corpus of reviews. In other words, we want identify words that strongly signal a review is either positive or negative. For example, we understand a word like "terrible" will mostly appear in negative rather than positive reviews. The naive Bayes model calculates probabilities such as $P(\text{terrible } | \text{ negative})$, the probability the review is negative given the word "terrible" appears in the text. Using these probabilities, we can derive a polarity score for each counted word,
# 
# $$
# \text{polarity} =  \log\left(\frac{P(\text{word } | \text{ positive})}{P(\text{word } | \text{ negative})}\right).
# $$ 
# 
# The polarity analysis is an example where a simpler model offers more explicability than a more complicated model. For this question, you are asked to determine the top twenty-five words with the largest positive **and** largest negative polarity, for a total of fifty words. For this analysis, you should:
# 
# 1. Use the naive Bayes model, `MultinomialNB`.
# 1. Use tf-idf weighting.
# 1. Remove stop words.
# 
# A trained naive Bayes model stores the log of the probabilities in the attribute `feature_log_prob_`. It is a NumPy array of shape (number of classes, the number of features). You will need the mapping between feature index to word. For this problem, you will use a different data set; it has been processed to only include reviews with one and five stars. You can download it below.

# In[29]:


get_ipython().run_cell_magic('bash', '', 'wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_one_and_five_star_reviews.json.gz -nc -P ./data')


# To avoid memory issue, we can delete the older data.

# In[30]:


del data, ratings


# In[31]:


import numpy as np
from sklearn.naive_bayes import MultinomialNB

with gzip.open("data/amazon_one_and_five_star_reviews.json.gz", "r") as f:
    data_polarity = [json.loads(line) for line in f]

ratings = [row['overall'] for row in data_polarity]


# In[32]:


pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings)


# In[33]:


# Get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

# Get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

# Collapse log probability into one row
polarity = log_prob[0,: ] - log_prob[1,: ]

# Combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))

# Get top and bottom of most_polar
n=25
most_polar = most_polar[:n] + most_polar[-n:]

# Get only terms from most_polar
top_50 = [term for score, term in most_polar]


# In[34]:


grader.score.nlp__most_polar(top_50)


# ## Question 5: Topic modeling [optional]
# 
# Topic modeling is the analysis of determining the key topics or themes in a corpus. With respect to machine learning, topic modeling is an unsupervised technique. One way to uncover the main topics in a corpus is to use [non-negative matrix factorization](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html). For this question, use non-negative matrix factorization to determine the top ten words for the first twenty topics. You should submit your answer as a list of lists. What topics exist in the reviews?

# In[36]:


from sklearn.decomposition import NMF
# nmf = NMF(n_components=20)
topic_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('dim', NMF(n_components=20))
])

topic_model.fit(data_polarity)


# In[37]:


nmf = topic_model.named_steps['dim']
index_to_token = topic_model.named_steps['vectorizer'].get_feature_names()


# In[39]:


nmf.components_.shape


# In[45]:


word_in_topics = []
for topic in nmf.components_:
    idx = topic.argsort()[-10:]
    top_words = [index_to_token[i] for i in idx]
    word_in_topics.append(top_words)


# In[57]:


word_in_topics[6]


# In[58]:


import pandas as pd


# In[63]:


df = pd.DataFrame(data=word_in_topics)


# In[64]:


df


# *Copyright &copy; 2019 The Data Incubator.  All rights reserved.*
