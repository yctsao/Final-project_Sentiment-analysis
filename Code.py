#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import json
import numpy as np
import os
import pandas as pd
import csv
import string
import utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


# In[2]:


def evaluation_metric(Truelabel, predictions, beta = 0):
    cm = confusion_matrix(Truelabel, predictions)
    classes = ['pos', 'neg']
    ex = pd.DataFrame(cm, columns=classes,index=classes)
    ex.index.name = "observed"
    
    
    eval = {}
    accuracy = ex.values.diagonal().sum() / ex.values.sum()
    precision = ex.values.diagonal() / ex.sum(axis=0)
    recall = ex.values.diagonal() / ex.sum(axis=1)
    f_score = (beta**2 + 1) * ((precision * recall) / ((beta**2 * precision) + recall))
    macro_f_score = f_score.mean(skipna=False)
    eval['accuracy'] = accuracy
    eval['precision'] = precision
    eval['recall'] = recall
    eval['f_score'] = f_score
    eval['macro_f_score'] = macro_f_score
    return pd.DataFrame.from_dict(eval)


# In[3]:


DATA_HOME = 'data'

REVIEWDATA_HOME = os.path.join(DATA_HOME, 'amazon_reviews_us_Camera_v1_00.tsv')

#test = os.path.join(DATA_HOME, 'cs224u-colors-bakeoff-data.csv')
# GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')


# In[4]:


START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
UNK_SYMBOL = "$UNK"


# In[5]:


df = pd.read_csv(REVIEWDATA_HOME, sep='\t',error_bad_lines=False, warn_bad_lines=False)


# In[6]:


df.columns.tolist()


# In[7]:


# Randomly select a small portion of the entire dataset for model implementation to save computation time
small_df = df.sample(frac=0.0001,random_state=1).reset_index(drop=True)
print(small_df.shape)
small_df.head(3)


# In[8]:


# Select features that may by useful for model training
features = ['product_id', 'star_rating', 'helpful_votes', 'total_votes', 'verified_purchase', 'review_headline', 'review_body']
small_df_selectedFeatures = small_df[features]

# Convert 5-star ratings into 2 (positive and negative) or 3 (positive, neutral and negative) classes
def starRating_to_PosNegClass(dataset, num_class):
    if num_class == 2:    
        # Negative reviews: 1-3 Stars = 0 ; Positive reviews: 4-5 Stars = 1
        dataset['pos_neg'] = [1 if x > 3 else 0 for x in dataset.star_rating]
    if num_class == 3: 
        # Negative reviews: 1-2 Stars = -1 ; Positive reviews: 4-5 Stars = 1 ; Neutral Reviews: 3 Stars = 0
        dataset['pos_neg_1'] = [1 if x > 3 else 0 for x in dataset.star_rating] 
        dataset['pos_neg_2'] = [-1 if x < 3 else 0 for x in dataset.star_rating] 
        dataset['pos_neg'] = (dataset['pos_neg_1'] + dataset['pos_neg_2'])
        dataset.drop(['pos_neg_1', 'pos_neg_2'], axis=1)
    return dataset


small_df_selectedFeatures = starRating_to_PosNegClass(small_df_selectedFeatures, num_class=2)
print('# of positive reviews and negative reviews: ')
small_df_selectedFeatures['pos_neg'].value_counts()


# In[9]:


small_df_selectedFeatures.head(5)


# In[10]:


# CountVectorizer -> Tokenizer + bags of words

def word_counts_featurizer(data):
    model = CountVectorizer()  # Convert a collection of text documents to a matrix of token counts
    X = model.fit_transform(data)
    print('# features: {}'.format(X.shape[1]))
    print ("Show some feature names : \n", model.get_feature_names()[::500])
    return X

def KNN(X_train, Y_train, X_test, Y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn_model_1 = knn.fit(X_train, Y_train)
    knn_prediction_train = knn_model_1.predict(X_train)
    knn_prediction_test = knn_model_1.predict(X_test)
    eval_train = evaluation_metric(small_Y_train, knn_prediction_train)
    eval_test = evaluation_metric(small_Y_test, knn_prediction_test)
    print('k-NN accuracy for test set: %f' % knn_model_1.score(X_test, Y_test))

    print (eval_train)
    print (eval_test)


small_X = word_counts_featurizer(small_df_selectedFeatures['review_body'])
small_Y = small_df_selectedFeatures['pos_neg']
small_X_train, small_X_test, small_Y_train, small_Y_test = train_test_split(small_X, small_Y, test_size=0.3, random_state=0)
KNN(small_X_train, small_Y_train, small_X_test, small_Y_test, n_neighbors=5)


# In[11]:


# Try with logistic regression and run 5 fold cross validation to find the best parameter for logistic regression
def fit_logistic_regression_with_crossvalidation(X, y):
    """A MaxEnt model of dataset with hyperparameter 
    cross-validation. Some notes:
        
    * 'fit_intercept': whether to include the class bias feature.
    * 'C': weight for the regularization term (smaller is more regularized).
    * 'penalty': type of regularization -- roughly, 'l1' ecourages small 
      sparse models, and 'l2' encourages the weights to conform to a 
      gaussian prior distribution.
    
    Other arguments can be cross-validated; see 
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.   
    
    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model instance, the best model found.
    
    """    
    basemod = LogisticRegression(
        fit_intercept=True, 
        solver='liblinear', 
        multi_class='auto')
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'C': [0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'penalty': ['l1','l2']}    
    best_mod = utils.fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid)
    
    return best_mod


# In[12]:


def best_model_Logistic_regression (small_X_train, small_Y_train, small_X_test, small_Y_test):
    best_mod = fit_logistic_regression_with_crossvalidation(small_X_train, small_Y_train)
    predictions_train = best_mod.predict(small_X_train)
    predictions_test = best_mod.predict(small_X_test)
    eval_train = evaluation_metric(small_Y_train, predictions_train)
    eval_test = evaluation_metric(small_Y_test, predictions_test)
    print("best_mod : \n", best_mod)
    print("train : \n", eval_train)
    print("test : \n", eval_test)


# In[13]:


best_mod = fit_logistic_regression_with_crossvalidation(small_X_train, small_Y_train)
best_model_Logistic_regression (small_X_train, small_Y_train, small_X_test, small_Y_test)


# In[ ]:


# train, test = train_test_split(df, test_size=0.3)
# print('training dataset: ', train.shape)
# print('test dataset: ', test.shape)

small_train, small_test = train_test_split(small_df_selectedFeatures, test_size=0.3, random_state=0)
print('small training dataset: ', small_train.shape)
print('smalle test dataset: ', small_test.shape)


# In[18]:


import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
import os


# In[19]:


#參照網路上例子的glove embedding
embeddings_index = {}
f = open('data/glove.840B.300d.txt', encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
    except ValueError:
       pass
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# In[41]:


#word_tokenize and glove embedding
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    #words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v.T


# In[42]:


small_X_g = small_df_selectedFeatures['review_body']
small_Y_g = small_df_selectedFeatures['pos_neg']
small_X_train_g, small_X_test_g, small_Y_train_g, small_Y_test_g = train_test_split(small_X_g, small_Y_g, test_size=0.3, random_state=0)


# In[43]:


#Transform all the data to glove and sent2vec
xtrain_glove = [sent2vec(x) for x in tqdm(small_X_train_g)]
xtest_glove = [sent2vec(x) for x in tqdm(small_X_test_g)]


# In[44]:


# logistic regression
clf = LogisticRegression()
clf.fit(xtrain_glove, small_Y_train)
predictions_train = clf.predict(xtrain_glove)
predictions_test = clf.predict(xtest_glove)
train = evaluation_metric(small_Y_train_g, predictions_train)
test = evaluation_metric(small_Y_test_g, predictions_test)
print(train)
print(test)


# In[45]:


# Try with bert embeddings


# In[20]:


from nltk.tokenize import TreebankWordTokenizer

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_example(s):
    return [START_SYMBOL] + bert_tokenizer.tokenize(s) + [END_SYMBOL]


# In[ ]:


tokenize_example(small_X_train_g[1])


# In[ ]:


def test(s):
    words = str(s).lower()
    words = tokenize_example(words)
    #words = [w for w in words if w.isalpha()]


# In[ ]:


#word_tokenize and glove embedding
def sent2vec(s):
    words = str(s).lower()
    words = tokenize_example(words)
    #words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v.T


# In[ ]:


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import transformers as ppb


# In[ ]:


import logging
logger = logging.getLogger()
logger.level = logging.ERROR


# In[ ]:


# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# In[ ]:


tokenized = small_X_train_g.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=511)))


# In[ ]:


#padding
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


# In[ ]:


np.array(padded).shape


# In[ ]:


attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


# In[ ]:


input_ids = torch.as_tensor(padded)
attention_mask = torch.as_tensor(attention_mask)
type(input_ids)


# In[ ]:


input_ids = torch.tensor(padded).long()
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)


# In[ ]:


features_train = last_hidden_states[0][:,0,:].numpy()
labels_train = small_Y_train_g


# In[ ]:


lr_clf = LogisticRegression()
lr_clf.fit(features_train, labels_train)
predictions_train = lr_clf.predict(features_train)


# In[ ]:


train = evaluation_metric(small_Y_train_g, predictions_train)
print(train)

