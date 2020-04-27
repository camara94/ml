
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('logstop', '')
get_ipython().run_line_magic('logstart', '-ortq ~/.logs/ml.py append')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


# In[4]:


from static_grader import grader


# # ML Miniproject
# ## Introduction
# 
# The objective of this miniproject is to exercise your ability to create effective machine learning models for making predictions. We will be working with nursing home inspection data from the United States, predicting which providers may be fined and for how much.
# 
# ## Scoring
# 
# In this miniproject you will often submit your model's `predict` or `predict_proba` method to the grader. The grader will assess the performance of your model using a scoring metric, comparing it against the score of a reference model. We will use the [average precision score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). If your model performs better than the reference solution, then you can score higher than 1.0.
# 
# **Note:** If you use an estimator that relies on random draws (like a `RandomForestClassifier`) you should set the `random_state=` to an integer so that your results are reproducible. 
# 
# ## Downloading the data
# 
# We can download the data set from Amazon S3:

# In[5]:


get_ipython().run_cell_magic('bash', '', 'mkdir data\nwget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data\nwget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data')


# We'll load the data into a Pandas DataFrame. Several columns will become target labels in future questions. Let's pop those columns out from the data, and drop related columns that are neither targets nor reasonable features (i.e. we don't wouldn't know how many times a facility denied payment before knowing whether it was fined).
# 
# The data has many columns. We have also provided a data dictionary.

# In[6]:


import numpy as np
import pandas as pd


# In[7]:


metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()


# In[8]:


data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')


# ## Question 1: state_model
# 
# A federal agency, Centers for Medicare and Medicaid Services (CMS), imposes regulations on nursing homes. However, nursing homes are inspected by state agencies for compliance with regulations, and fines for violations can vary widely between states.
# 
# Let's develop a very simple initial model to predict the amount of fines a nursing home might expect to pay based on its location. Fill in the class definition of the custom estimator, `StateMeanEstimator`, below.

# In[9]:


from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = (y.groupby(X[self.gb_col])
                                       .mean().to_dict())
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        # Return a list of predicted penalties based on group of samples in X
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return [self.group_averages.get(row, self.global_avg) for row in X[self.gb_col]]


# After filling in class definition, we can create an instance of the estimator and fit it to the data.

# In[10]:


from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)


# Next we should test that our predict method works.

# In[11]:


state_model.predict(data.sample(5))


# However, what if we have data from a nursing home in a state (or territory) of the US which is not in the training data?

# In[12]:


state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))


# Make sure your model can handle this possibility before submitting your model's predict method to the grader.

# In[13]:


grader.score.ml__state_model(state_model.predict)


# ## Question 2: simple_features_model
# 
# Nursing homes vary greatly in their business characteristics. Some are owned by the government or non-profits while others are run for profit. Some house a few dozen residents while others house hundreds. Some are located within hospitals and may work with more vulnerable populations. We will try to predict which facilities are fined based on their business characteristics.
# 
# We'll begin with columns in our DataFrame containing numeric and boolean features. Some of the rows contain null values; estimators cannot handle null values so these must be imputed or dropped. We will create a `Pipeline` containing transformers that process these features, followed by an estimator.
# 
# **Note:** When the grader checks your answer, it passes a list of dictionaries to the `predict` or `predict_proba` method of your estimator, not a DataFrame. This means that your model must work with both data types. For this reason, we've provided a custom `ColumnSelectTransformer` for you to use instead `scikit-learn`'s own `ColumnTransformer`.

# In[14]:


from sklearn.impute import SimpleImputer
simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])


# In[15]:


len(simple_cols)


# In[16]:


pd.DataFrame(simple_features.fit_transform(data)).info()


# In[17]:


pd.DataFrame(simple_features.fit_transform(data), columns=simple_cols).head()


# In[18]:


data[simple_cols].head()


# **Note:** The assertion below assumes the output of `noncategorical_features.fit_transform` is a `ndarray`, not a `DataFrame`.)

# In[19]:


assert data['RESTOT'].isnull().sum() > 0


# In[20]:


assert not np.isnan(simple_features.fit_transform(data)).any()


# Now combine the `simple_features` pipeline with an estimator in a new pipeline. Fit `simple_features_model` to the data and submit `simple_features_model.predict_proba` to the grader. You may wish to use cross-validation to tune the hyperparameters of your model.

# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[22]:


simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    #('predictor', LogisticRegression(solver='lbfgs'))
    #('predictor', RandomForestClassifier()) score:  0.597184769712005
    #('predictor', SVC(probability=True)) score:  0.4867148455758089
    ('predictor', RandomForestClassifier(n_estimators=100, max_depth=10))
])


# In[23]:


simple_features_model.fit(data, fine_counts > 0)


# In[24]:


def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))


# ## Question 3: categorical_features

# The `'OWNERSHIP'` and `'CERTIFICATION'` columns contain categorical data. We will have to encode the categorical data into numerical features before we pass them to an estimator. Construct one or more pipelines for this purpose. Transformers such as [LabelEncoder](https://scikit-learn.org/0.19/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) and [OneHotEncoder](https://scikit-learn.org/0.19/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) may be useful, but you may also want to define your own transformers.
# 
# If you used more than one `Pipeline`, combine them with a `FeatureUnion`. As in Question 2, we will combine this with an estimator, fit it, and submit the `predict_proba` method to the grader.

# In[25]:


# data.OWNERSHIP.value_counts()
data.CERTIFICATION.value_counts()


# In[26]:


from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder


# In[27]:


owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP']) ),
    ('ohe', OneHotEncoder(categories='auto', sparse=False) )
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder(categories='auto', sparse=False) )
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
 ])


# In[28]:


pd.DataFrame(cert_onehot.fit_transform(data),
            columns=cert_onehot.named_steps['ohe'].categories_[0]).head(10)


# In[29]:


assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()


# In[30]:


from sklearn.naive_bayes import MultinomialNB


# As in the previous question, create a model using the `categorical_features`, fit it to the data, and submit its `predict_proba` method to the grader.

# In[31]:


categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    #('classifier', RandomForestClassifier(n_estimators=110, max_depth=12))
    ('classifier', MultinomialNB(alpha=0.01))
])


# In[32]:


categorical_features_model.fit(data, fine_counts > 0)


# In[33]:


grader.score.ml__categorical_features(positive_probability(categorical_features_model))


# ## Question 4: business_model

# Finally, we'll combine `simple_features` and `categorical_features` in a `FeatureUnion`, followed by an estimator in a `Pipeline`. You may want to optimize the hyperparameters of your estimator using cross-validation or try engineering new features (e.g. see [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)). When you've assembled and trained your model, pass the `predict_proba` method to the grader.

# In[34]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression


# In[41]:


business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])


# In[42]:



business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])


# In[65]:


business_model.fit(data, fine_counts > 0)


# In[66]:


grader.score.ml__business_model(positive_probability(business_model))


# ## Question 5: survey_results

# Surveys reveal safety and health deficiencies at nursing homes that may indicate risk for incidents (and penalties). CMS routinely makes surveys of nursing homes. Build a model that combines the `business_features` of each facility with its cycle 1 survey results, as well as the time between the cycle 1 and cycle 2 survey to predict the cycle 2 total score.
# 
# First, let's create a transformer to calculate the difference in time between the cycle 1 and cycle 2 surveys.

# In[67]:


data.info()


# In[68]:


test_df = data[['CYCLE_1_SURVEY_DATE','CYCLE_2_SURVEY_DATE']]


# In[69]:


test_df['CYCLE_1_SURVEY_DATE'] = pd.to_datetime(test_df['CYCLE_1_SURVEY_DATE'])
test_df['CYCLE_2_SURVEY_DATE'] = pd.to_datetime(test_df['CYCLE_2_SURVEY_DATE'])


# In[70]:


test_df['delta'] = test_df['CYCLE_1_SURVEY_DATE'] - test_df['CYCLE_2_SURVEY_DATE']


# In[71]:


test_df.delta.head().values


# In[72]:


test_df.head()


# In[73]:


#  test_df['delta'] = test_df['delta'].apply(lambda x: x.days)


# In[74]:


class TimedeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col):
        self.t1_col = t1_col
        self.t2_col = t2_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        results = (pd.to_datetime(X[ self.t1_col]) - pd.to_datetime(X[ self.t2_col]))
        results = results.apply(lambda x: x.days).values.reshape(-1, 1)
        return results;


# In[75]:


cycle_1_date = 'CYCLE_1_SURVEY_DATE'
cycle_2_date = 'CYCLE_2_SURVEY_DATE'
time_feature = TimedeltaTransformer(cycle_1_date, cycle_2_date)


# In[76]:


time_feature.fit_transform(data.head())


# In the cell below we'll collect the cycle 1 survey features.

# In[77]:


cycle_1_cols = ['CYCLE_1_DEFS', 'CYCLE_1_NFROMDEFS', 'CYCLE_1_NFROMCOMP',
                'CYCLE_1_DEFS_SCORE', 'CYCLE_1_NUMREVIS',
                'CYCLE_1_REVISIT_SCORE', 'CYCLE_1_TOTAL_SCORE']
cycle_1_features = ColumnSelectTransformer(cycle_1_cols)


# In[117]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD 
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(Lasso(max_iter=1000), 
                 param_grid={'alpha':np.arange(0,3.5,0.5)},
                 cv=5,
                 n_jobs=4,
                 verbose=1
                 )

survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('decomp', TruncatedSVD(40)),
    ('gs', gs)
    #('predictor', Lasso(alpha=3, max_iter=1000))
])


# In[118]:


data.shape


# In[119]:


survey_model.fit(data, cycle_2_score.astype(int))


# In[120]:


grader.score.ml__survey_model(survey_model.predict)


# *Copyright &copy; 2019 The Data Incubator.  All rights reserved.*
