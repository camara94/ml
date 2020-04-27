

```python
%logstop
%logstart -ortq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
```


```python
from static_grader import grader
```

# ML Miniproject
## Introduction

The objective of this miniproject is to exercise your ability to create effective machine learning models for making predictions. We will be working with nursing home inspection data from the United States, predicting which providers may be fined and for how much.

## Scoring

In this miniproject you will often submit your model's `predict` or `predict_proba` method to the grader. The grader will assess the performance of your model using a scoring metric, comparing it against the score of a reference model. We will use the [average precision score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). If your model performs better than the reference solution, then you can score higher than 1.0.

**Note:** If you use an estimator that relies on random draws (like a `RandomForestClassifier`) you should set the `random_state=` to an integer so that your results are reproducible. 

## Downloading the data

We can download the data set from Amazon S3:


```bash
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data
```

    mkdir: cannot create directory ‘data’: File exists
    File ‘./ml-data/providers-train.csv’ already there; not retrieving.
    
    File ‘./ml-data/providers-metadata.csv’ already there; not retrieving.
    


We'll load the data into a Pandas DataFrame. Several columns will become target labels in future questions. Let's pop those columns out from the data, and drop related columns that are neither targets nor reasonable features (i.e. we don't wouldn't know how many times a facility denied payment before knowing whether it was fined).

The data has many columns. We have also provided a data dictionary.


```python
import numpy as np
import pandas as pd
```


```python
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Label</th>
      <th>Description</th>
      <th>Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PROVNUM</td>
      <td>Federal Provider Number</td>
      <td>Federal Provider Number</td>
      <td>6 alphanumeric characters</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PROVNAME</td>
      <td>Provider Name</td>
      <td>Provider Name</td>
      <td>text</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADDRESS</td>
      <td>Provider Address</td>
      <td>Provider Address</td>
      <td>text</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CITY</td>
      <td>Provider City</td>
      <td>Provider City</td>
      <td>text</td>
    </tr>
    <tr>
      <th>4</th>
      <td>STATE</td>
      <td>Provider State</td>
      <td>Provider State</td>
      <td>2-character postal abbreviation</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')
```

## Question 1: state_model

A federal agency, Centers for Medicare and Medicaid Services (CMS), imposes regulations on nursing homes. However, nursing homes are inspected by state agencies for compliance with regulations, and fines for violations can vary widely between states.

Let's develop a very simple initial model to predict the amount of fines a nursing home might expect to pay based on its location. Fill in the class definition of the custom estimator, `StateMeanEstimator`, below.


```python
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
```

After filling in class definition, we can create an instance of the estimator and fit it to the data.


```python
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)
```




    Pipeline(memory=None, steps=[('sme', GroupMeanEstimator(gb_col='STATE'))],
             verbose=False)



Next we should test that our predict method works.


```python
state_model.predict(data.sample(5))
```




    [3490.756838905775,
     2213.51526032316,
     29459.975,
     6634.197226502311,
     8214.822977725675]



However, what if we have data from a nursing home in a state (or territory) of the US which is not in the training data?


```python
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))
```




    [14969.857687877915]



Make sure your model can handle this possibility before submitting your model's predict method to the grader.


```python
grader.score.ml__state_model(state_model.predict)
```

    ==================
    Your score:  0.9999999999999999
    ==================


## Question 2: simple_features_model

Nursing homes vary greatly in their business characteristics. Some are owned by the government or non-profits while others are run for profit. Some house a few dozen residents while others house hundreds. Some are located within hospitals and may work with more vulnerable populations. We will try to predict which facilities are fined based on their business characteristics.

We'll begin with columns in our DataFrame containing numeric and boolean features. Some of the rows contain null values; estimators cannot handle null values so these must be imputed or dropped. We will create a `Pipeline` containing transformers that process these features, followed by an estimator.

**Note:** When the grader checks your answer, it passes a list of dictionaries to the `predict` or `predict_proba` method of your estimator, not a DataFrame. This means that your model must work with both data types. For this reason, we've provided a custom `ColumnSelectTransformer` for you to use instead `scikit-learn`'s own `ColumnTransformer`.


```python
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
```


```python
len(simple_cols)
```




    9




```python
pd.DataFrame(simple_features.fit_transform(data)).info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13892 entries, 0 to 13891
    Data columns (total 9 columns):
    0    13892 non-null float64
    1    13892 non-null float64
    2    13892 non-null float64
    3    13892 non-null float64
    4    13892 non-null float64
    5    13892 non-null float64
    6    13892 non-null float64
    7    13892 non-null float64
    8    13892 non-null float64
    dtypes: float64(9)
    memory usage: 976.9 KB



```python
pd.DataFrame(simple_features.fit_transform(data), columns=simple_cols).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BEDCERT</th>
      <th>RESTOT</th>
      <th>INHOSP</th>
      <th>CCRC_FACIL</th>
      <th>SFF</th>
      <th>CHOW_LAST_12MOS</th>
      <th>SPRINKLER_STATUS</th>
      <th>EXP_TOTAL</th>
      <th>ADJ_TOTAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85.0</td>
      <td>74.200000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.212187</td>
      <td>3.859121</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50.0</td>
      <td>86.760469</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.212187</td>
      <td>3.859121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92.0</td>
      <td>79.800000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.080150</td>
      <td>3.830260</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103.0</td>
      <td>98.100000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.839380</td>
      <td>3.957090</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149.0</td>
      <td>119.700000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.155540</td>
      <td>4.078660</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[simple_cols].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BEDCERT</th>
      <th>RESTOT</th>
      <th>INHOSP</th>
      <th>CCRC_FACIL</th>
      <th>SFF</th>
      <th>CHOW_LAST_12MOS</th>
      <th>SPRINKLER_STATUS</th>
      <th>EXP_TOTAL</th>
      <th>ADJ_TOTAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>85</td>
      <td>74.2</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92</td>
      <td>79.8</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>3.08015</td>
      <td>3.83026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103</td>
      <td>98.1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>2.83938</td>
      <td>3.95709</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>119.7</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>3.15554</td>
      <td>4.07866</td>
    </tr>
  </tbody>
</table>
</div>



**Note:** The assertion below assumes the output of `noncategorical_features.fit_transform` is a `ndarray`, not a `DataFrame`.)


```python
assert data['RESTOT'].isnull().sum() > 0
```


```python
assert not np.isnan(simple_features.fit_transform(data)).any()
```

Now combine the `simple_features` pipeline with an estimator in a new pipeline. Fit `simple_features_model` to the data and submit `simple_features_model.predict_proba` to the grader. You may wish to use cross-validation to tune the hyperparameters of your model.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
```


```python
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    #('predictor', LogisticRegression(solver='lbfgs'))
    #('predictor', RandomForestClassifier()) score:  0.597184769712005
    #('predictor', SVC(probability=True)) score:  0.4867148455758089
    ('predictor', RandomForestClassifier(n_estimators=100, max_depth=10))
])
```


```python
simple_features_model.fit(data, fine_counts > 0)
```




    Pipeline(memory=None,
             steps=[('simple',
                     Pipeline(memory=None,
                              steps=[('cst',
                                      ColumnSelectTransformer(columns=['BEDCERT',
                                                                       'RESTOT',
                                                                       'INHOSP',
                                                                       'CCRC_FACIL',
                                                                       'SFF',
                                                                       'CHOW_LAST_12MOS',
                                                                       'SPRINKLER_STATUS',
                                                                       'EXP_TOTAL',
                                                                       'ADJ_TOTAL'])),
                                     ('imputer',
                                      SimpleImputer(add_indicator=False, copy=True,
                                                    fill_value=None,
                                                    missing_values=nan,
                                                    strategy='mean', verbose=0))],
                              verbose=False)...
                     RandomForestClassifier(bootstrap=True, class_weight=None,
                                            criterion='gini', max_depth=10,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=100, n_jobs=None,
                                            oob_score=False, random_state=None,
                                            verbose=0, warm_start=False))],
             verbose=False)




```python
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))
```

    ==================
    Your score:  1.0462283824111012
    ==================


## Question 3: categorical_features

The `'OWNERSHIP'` and `'CERTIFICATION'` columns contain categorical data. We will have to encode the categorical data into numerical features before we pass them to an estimator. Construct one or more pipelines for this purpose. Transformers such as [LabelEncoder](https://scikit-learn.org/0.19/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) and [OneHotEncoder](https://scikit-learn.org/0.19/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) may be useful, but you may also want to define your own transformers.

If you used more than one `Pipeline`, combine them with a `FeatureUnion`. As in Question 2, we will combine this with an estimator, fit it, and submit the `predict_proba` method to the grader.


```python
# data.OWNERSHIP.value_counts()
data.CERTIFICATION.value_counts()
```




    Medicare and Medicaid    12942
    Medicare                   634
    Medicaid                   316
    Name: CERTIFICATION, dtype: int64




```python
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
```


```python
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
```


```python
pd.DataFrame(cert_onehot.fit_transform(data),
            columns=cert_onehot.named_steps['ohe'].categories_[0]).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Medicaid</th>
      <th>Medicare</th>
      <th>Medicare and Medicaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()
```


```python
from sklearn.naive_bayes import MultinomialNB
```

As in the previous question, create a model using the `categorical_features`, fit it to the data, and submit its `predict_proba` method to the grader.


```python
categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    #('classifier', RandomForestClassifier(n_estimators=110, max_depth=12))
    ('classifier', MultinomialNB(alpha=0.01))
])
```


```python
categorical_features_model.fit(data, fine_counts > 0)
```




    Pipeline(memory=None,
             steps=[('categorical',
                     FeatureUnion(n_jobs=None,
                                  transformer_list=[('owner',
                                                     Pipeline(memory=None,
                                                              steps=[('cst',
                                                                      ColumnSelectTransformer(columns=['OWNERSHIP'])),
                                                                     ('ohe',
                                                                      OneHotEncoder(categorical_features=None,
                                                                                    categories='auto',
                                                                                    drop=None,
                                                                                    dtype=<class 'numpy.float64'>,
                                                                                    handle_unknown='error',
                                                                                    n_values=None,
                                                                                    sparse=False))],
                                                              verbose=False))...
                                                                      ColumnSelectTransformer(columns=['CERTIFICATION'])),
                                                                     ('ohe',
                                                                      OneHotEncoder(categorical_features=None,
                                                                                    categories='auto',
                                                                                    drop=None,
                                                                                    dtype=<class 'numpy.float64'>,
                                                                                    handle_unknown='error',
                                                                                    n_values=None,
                                                                                    sparse=False))],
                                                              verbose=False))],
                                  transformer_weights=None, verbose=False)),
                    ('classifier',
                     MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))],
             verbose=False)




```python
grader.score.ml__categorical_features(positive_probability(categorical_features_model))
```

    ==================
    Your score:  0.9747736478437008
    ==================


## Question 4: business_model

Finally, we'll combine `simple_features` and `categorical_features` in a `FeatureUnion`, followed by an estimator in a `Pipeline`. You may want to optimize the hyperparameters of your estimator using cross-validation or try engineering new features (e.g. see [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)). When you've assembled and trained your model, pass the `predict_proba` method to the grader.


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
```


```python
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])
```


```python

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])
```


```python
business_model.fit(data, fine_counts > 0)
```

    /opt/conda/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    Pipeline(memory=None,
             steps=[('features',
                     FeatureUnion(n_jobs=None,
                                  transformer_list=[('simple',
                                                     Pipeline(memory=None,
                                                              steps=[('cst',
                                                                      ColumnSelectTransformer(columns=['BEDCERT',
                                                                                                       'RESTOT',
                                                                                                       'INHOSP',
                                                                                                       'CCRC_FACIL',
                                                                                                       'SFF',
                                                                                                       'CHOW_LAST_12MOS',
                                                                                                       'SPRINKLER_STATUS',
                                                                                                       'EXP_TOTAL',
                                                                                                       'ADJ_TOTAL'])),
                                                                     ('imputer',
                                                                      SimpleImputer(add_indicator=False,
                                                                                    copy=True,
                                                                                    fill_value=None,
                                                                                    missing...
                                  transformer_weights=None, verbose=False)),
                    ('poly',
                     PolynomialFeatures(degree=2, include_bias=True,
                                        interaction_only=False, order='C')),
                    ('lr',
                     LogisticRegression(C=1.0, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='warn', n_jobs=None,
                                        penalty='l2', random_state=None,
                                        solver='warn', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)




```python
grader.score.ml__business_model(positive_probability(business_model))
```

    ==================
    Your score:  0.9914203514059667
    ==================


## Question 5: survey_results

Surveys reveal safety and health deficiencies at nursing homes that may indicate risk for incidents (and penalties). CMS routinely makes surveys of nursing homes. Build a model that combines the `business_features` of each facility with its cycle 1 survey results, as well as the time between the cycle 1 and cycle 2 survey to predict the cycle 2 total score.

First, let's create a transformer to calculate the difference in time between the cycle 1 and cycle 2 surveys.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13892 entries, 0 to 13891
    Data columns (total 29 columns):
    PROVNUM                  13892 non-null object
    PROVNAME                 13892 non-null object
    ADDRESS                  13892 non-null object
    CITY                     13892 non-null object
    STATE                    13892 non-null object
    ZIP                      13892 non-null int64
    PHONE                    13892 non-null int64
    COUNTY_SSA               13892 non-null int64
    COUNTY_NAME              13892 non-null object
    BEDCERT                  13892 non-null int64
    RESTOT                   13483 non-null float64
    INHOSP                   13892 non-null bool
    CCRC_FACIL               13892 non-null bool
    SFF                      13892 non-null bool
    CHOW_LAST_12MOS          13892 non-null bool
    SPRINKLER_STATUS         13892 non-null bool
    EXP_TOTAL                13104 non-null float64
    ADJ_TOTAL                13104 non-null float64
    OWNERSHIP                13892 non-null object
    CERTIFICATION            13892 non-null object
    CYCLE_1_DEFS             13892 non-null int64
    CYCLE_1_NFROMDEFS        13892 non-null int64
    CYCLE_1_NFROMCOMP        13892 non-null int64
    CYCLE_1_DEFS_SCORE       13892 non-null int64
    CYCLE_1_NUMREVIS         13892 non-null int64
    CYCLE_1_REVISIT_SCORE    13892 non-null int64
    CYCLE_1_TOTAL_SCORE      13892 non-null int64
    CYCLE_1_SURVEY_DATE      13892 non-null object
    CYCLE_2_SURVEY_DATE      13892 non-null object
    dtypes: bool(5), float64(3), int64(11), object(10)
    memory usage: 2.6+ MB



```python
test_df = data[['CYCLE_1_SURVEY_DATE','CYCLE_2_SURVEY_DATE']]
```


```python
test_df['CYCLE_1_SURVEY_DATE'] = pd.to_datetime(test_df['CYCLE_1_SURVEY_DATE'])
test_df['CYCLE_2_SURVEY_DATE'] = pd.to_datetime(test_df['CYCLE_2_SURVEY_DATE'])
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
test_df['delta'] = test_df['CYCLE_1_SURVEY_DATE'] - test_df['CYCLE_2_SURVEY_DATE']
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
test_df.delta.head().values
```




    array([27216000000000000, 35078400000000000, 25488000000000000,
           33868800000000000, 33264000000000000], dtype='timedelta64[ns]')




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CYCLE_1_SURVEY_DATE</th>
      <th>CYCLE_2_SURVEY_DATE</th>
      <th>delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-04-06</td>
      <td>2016-05-26</td>
      <td>315 days</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-03-16</td>
      <td>2016-02-04</td>
      <td>406 days</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-10-20</td>
      <td>2015-12-30</td>
      <td>295 days</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-03-09</td>
      <td>2016-02-11</td>
      <td>392 days</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-06-01</td>
      <td>2016-05-12</td>
      <td>385 days</td>
    </tr>
  </tbody>
</table>
</div>




```python
#  test_df['delta'] = test_df['delta'].apply(lambda x: x.days)
```


```python
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
```


```python
cycle_1_date = 'CYCLE_1_SURVEY_DATE'
cycle_2_date = 'CYCLE_2_SURVEY_DATE'
time_feature = TimedeltaTransformer(cycle_1_date, cycle_2_date)
```


```python
time_feature.fit_transform(data.head())
```




    array([[315],
           [406],
           [295],
           [392],
           [385]])



In the cell below we'll collect the cycle 1 survey features.


```python
cycle_1_cols = ['CYCLE_1_DEFS', 'CYCLE_1_NFROMDEFS', 'CYCLE_1_NFROMCOMP',
                'CYCLE_1_DEFS_SCORE', 'CYCLE_1_NUMREVIS',
                'CYCLE_1_REVISIT_SCORE', 'CYCLE_1_TOTAL_SCORE']
cycle_1_features = ColumnSelectTransformer(cycle_1_cols)
```


```python
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
```


```python
data.shape
```




    (13892, 29)




```python
survey_model.fit(data, cycle_2_score.astype(int))
```

    Fitting 5 folds for each of 7 candidates, totalling 35 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  35 out of  35 | elapsed:    7.4s finished





    Pipeline(memory=None,
             steps=[('features',
                     FeatureUnion(n_jobs=None,
                                  transformer_list=[('business',
                                                     FeatureUnion(n_jobs=None,
                                                                  transformer_list=[('simple',
                                                                                     Pipeline(memory=None,
                                                                                              steps=[('cst',
                                                                                                      ColumnSelectTransformer(columns=['BEDCERT',
                                                                                                                                       'RESTOT',
                                                                                                                                       'INHOSP',
                                                                                                                                       'CCRC_FACIL',
                                                                                                                                       'SFF',
                                                                                                                                       'CHOW_LAST_12MOS',
                                                                                                                                       'SPRINKLER_STATUS',
                                                                                                                                       'EXP_TOTAL',
                                                                                                                                       'ADJ_TOTAL'])),
                                                                                                     ('imputer',
                                                                                                      SimpleImpute...
                                  estimator=Lasso(alpha=1.0, copy_X=True,
                                                  fit_intercept=True, max_iter=1000,
                                                  normalize=False, positive=False,
                                                  precompute=False,
                                                  random_state=None,
                                                  selection='cyclic', tol=0.0001,
                                                  warm_start=False),
                                  iid='warn', n_jobs=4,
                                  param_grid={'alpha': array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. ])},
                                  pre_dispatch='2*n_jobs', refit=True,
                                  return_train_score=False, scoring=None,
                                  verbose=1))],
             verbose=False)




```python
grader.score.ml__survey_model(survey_model.predict)
```

    ==================
    Your score:  1.163796091675108
    ==================


*Copyright &copy; 2019 The Data Incubator.  All rights reserved.*
