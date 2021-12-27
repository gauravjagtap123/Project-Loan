
# 

# In[157]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report,confusion_matrix
import scipy.optimize as opt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[ ]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[136]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[137]:


df.shape


# ### Convert to date time object
# 

# In[138]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[139]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[ ]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[140]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[141]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[142]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[143]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[144]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[145]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[13]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[14]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[15]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[16]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[17]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[18]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[69]:


K = 6
neigh6 = KNeighborsClassifier(n_neighbors = K).fit(X_train,y_train)
neigh6
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ",metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ",metrics.accuracy_score(y_test, yhat6))

print("Avg F1-score: %.4f" % f1_score(y_test, yhat6, average='weighted'))


# In[75]:


from sklearn.metrics import jaccard_score


# In[76]:


jaccard_KNN_score = jaccard_score(y_test, yhat6, average = None)
jaccard_KNN_score = jaccard_KNN_score[1]
jaccard_KNN_score = float("{:.2f}".format(jaccard_KNN_score))
jaccard_KNN_score


# In[77]:


from sklearn.metrics import confusion_matrix


# In[82]:


cm = confusion_matrix(y_test, yhat6)
sns.heatmap(cm, annot=True)


# #### Let's calculate accuracy for different values of K

# In[37]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    ##Train model and predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1]= metrics.accuracy_score(y_test,yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# # Decision Tree
# 

# In[42]:


dt_train_prediction=DecisionTreeClassifier(max_depth=4, criterion = 'entropy')
dt_train_prediction.fit(X_train, y_train)
dt_yhat=dt_train_prediction.predict(X_test)


# In[83]:


print('Accuracy score of the Decision.Tree model is {}'.format(accuracy_score(y_test,dt_yhat)))
print("Avg F1-score: %.4f" % f1_score(y_test, yhat6, average='weighted'))


# In[170]:


DTree_Jaccard = jaccard_score(y_test, dt_yhat, average = None)
DTree_Jaccard = DTree_Jaccard[1]
DTree_Jaccard = float("{:.2f}".format(DTree_Jaccard))
DTree_Jaccard


# In[85]:


cm = confusion_matrix(y_test,dt_yhat )
sns.heatmap(cm, annot=True)


# # Support Vector Machine
# 

# In[90]:


from sklearn.svm import SVC
my_SVM = SVC(kernel = 'rbf')
my_SVM.fit(X_train, y_train)


# In[91]:


y_predict_svm = my_SVM.predict(X_test)
y_predict_svm[0:5]


# In[92]:


print("Training set accuracy : ", accuracy_score(y_train, my_SVM.predict(X_train)))
print("Testing set accuracy : ", accuracy_score(y_test, my_SVM.predict(X_test)))


# In[93]:


jaccard_score_SVM = jaccard_score(y_test, y_predict_svm, average = None)
jaccard_score_SVM = jaccard_score_SVM[1]
jaccard_score_SVM


# In[94]:


cm = confusion_matrix(y_test, y_predict_svm)
sns.heatmap(cm, annot=True)


# # Logistic Regression
# 

# In[96]:


X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train_LR.shape,  y_train_LR.shape)
print ('Test set:', X_test_LR.shape,  y_test_LR.shape)


# In[97]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_LR, y_train_LR)
print(LR)


# In[98]:


y_predict_LR = LR.predict(X_test_LR)
y_predict_LR[0:5]


# In[99]:


y_predict_LR_prob = LR.predict_proba(X_test_LR)
print(y_predict_LR_prob)


# In[100]:


jaccard_score_LR = jaccard_score(y_test_LR, y_predict_LR, average = None)
jaccard_score_LR = jaccard_score_LR[1]
jaccard_score_LR = float("{:.2f}".format(jaccard_score_LR))
jaccard_score_LR


# In[ ]:





# # Model Evaluation using Test set
# 

# In[101]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:
# 

# In[102]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation
# 

# In[151]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[152]:


test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
test_df['dayofweek'] = df['effective_date'].dt.dayofweek
test_df.head()


# In[153]:


test_df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.head()


# In[154]:


test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[155]:


test_df['Gender'].replace(to_replace=['male','female'],value = [1,0],inplace=True)
test_df.head()


# In[158]:


test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_X = StandardScaler().fit(test_Feature).transform(test_Feature)
test_X[0:5]


# In[164]:


test_y = test_df['loan_status'].values
test_y[0:5]


# In[166]:


# K Nearest Neighbor (KNN)
test_yhat= neigh6.predict(test_X)

# Calculating the Jaccard Score for KNN
KNN_Jaccard = jaccard_score(test_y, test_yhat, average=None)
KNN_Jaccard = KNN_Jaccard[1]
KNN_jaccard = float("{:.2f}".format(KNN_Jaccard))
print("KNN - jaccard accuracy = " , KNN_Jaccard)

# f1_score
KNN_f1_score = f1_score(test_y, test_yhat, average='weighted') 
KNN_f1_score = float("{:.2f}".format(KNN_f1_score))
print("KNN - f1 score accuracy = " , KNN_f1_score)


# In[175]:


# Decision Tree
test_yhat=dt_train_prediction.predict(X_test)

DTree_Jaccard_score = jaccard_score(y_test, dt_yhat, average = None)
DTree_Jaccard_score = DTree_Jaccard_score[1]
DTree_Jaccard_score = float("{:.2f}".format(DTree_Jaccard_score))
print("Decision Tree - jaccard accuracy = " , DTree_Jaccard_score)

print("Avg F1-score: %.4f" % f1_score(y_test, yhat6, average='weighted'))


# In[174]:


# Support Vector Machine
test_yhat = my_SVM.predict(test_X)

# jaccard
SVM_Jaccard = jaccard_score(test_y, test_yhat, average=None)
SVM_Jaccard = SVM_Jaccard[1]
SVM_Jaccard = float("{:.2f}".format(SVM_Jaccard))
print("Support Vector Machine - jaccard accuracy = " , SVM_Jaccard)

# f1_score
SVM_f1_score = f1_score(test_y, test_yhat, average='weighted') 
SVM_f1_score = float("{:.2f}".format(SVM_f1_score))
print("Support Vector Machine - f1 score accuracy = " , SVM_f1_score)


# In[176]:


#Logistic Regression
test_yhat = LR.predict(test_X)
test_yhat_prob = LR.predict_proba(test_X)

# jaccard
LGR_Jaccard = jaccard_score(test_y, test_yhat, average=None)
LGR_Jaccard = LGR_Jaccard[1]
LGR_Jaccard = float("{:.2f}".format(LGR_Jaccard))
print("Logistic Regression - jaccard accuracy = " , LGR_Jaccard)

# f1_score
LGR_f1_score = f1_score(test_y, test_yhat, average='weighted')
LGR_f1_score = float("{:.2f}".format(LGR_f1_score))
print("Logistic Regression - f1 score accuracy = " , LGR_f1_score)

LGR_log_loss = log_loss(test_y, test_yhat_prob)
LGR_log_loss = float("{:.2f}".format(LGR_log_loss))
print("Logistic Regression - log loss = " , LGR_log_loss)


# In[180]:


# Print the Accuracy Report
#col_names = ['Algorithm', 'Jaccard', 'F1-score', 'LogLoss']
algorithm_list = ['KNN', 'Decision Tree', 'SVM', 'LoisticRegression']
Jaccard_list = [KNN_Jaccard, DTree_Jaccard_score, SVM_Jaccard, LGR_Jaccard]
F1_score_list = [KNN_f1_score, DTree_f1_score, SVM_f1_score, LGR_f1_score]
LogLoss_list = ['NA', 'NA', 'NA', LGR_log_loss]

df = pd.DataFrame(list(zip(algorithm_list, Jaccard_list, F1_score_list, LogLoss_list)),
              columns=['Algorithm','Jaccard', 'F1-score', 'LogLoss'])

df.set_index('Algorithm', inplace = True)
df








