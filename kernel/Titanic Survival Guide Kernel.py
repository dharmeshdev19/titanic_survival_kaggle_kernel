
# coding: utf-8

# # Machine Learning Approach with Python
# 
# This notebook covers the basic Machine Learning process in Python step-by-step.
# 
# ## **Table of Contents:**
# * Introduction
# * Breif History of RMS Titanic
# * Import Libraries
# * Loading the Data
# * Data Exploration/Analysis
# * Data Preprocessing
#     - Missing Data
#     - Converting Features
#     - Creating Categories
#     - Creating new Features
# * Building Machine Learning Models
#     - Training 8 different models
#     - Which is the best model ?
#     - K-Fold Cross Validation
# * Random Forest 
#     - What is Random Forest ?
#     - Feature importance
#     - Hyperparameter Tuning   
# * Further Evaluation 
#     - Confusion Matrix
#     - Precision and Recall 
#     - F-Score
#     - Precision Recall Curve
#     - ROC AUC Curve
#     - ROC AUC Score
# * Submission
# * Summary

# ## **Introduction**
# 
# In this kernel, I will go through the whole process of creating several machine learning models on the famous Titanic dataset, which is used by many people as beginner guide for getting started with Data Science / Machine Learning. It provides information on the fate of passengers on the Titanic, summarized according to economic status (class), sex, age, and survival. In this challenge, we are asked to predict whether a passenger on the Titanic would have been survived or not. Let us go through step by step process to match our prediction with the actual result.

# ## **Breif History of RMS Titanic**
# 
# RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. The RMS Titanic was the largest ship afloat at the time it entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. The Titanic was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, her architect, died in the disaster.

# ## **Import Libraries**
# *Note: As this is a step by step tutorial we will import the model based libraries later in the tutorial*

# In[1]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
sns.set(font_scale=1)
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style


# ## **Loading the Data**

# #### Load the train data

# In[2]:


# This creates a pandas dataframe and assigns it to the train variable.
train_df = pd.read_csv("../input/train.csv")


# In[3]:


# This creates a pandas dataframe and assigns it to the test variable.
test_df = pd.read_csv("../input/test.csv")


# ## **Data Exploration/Analysis**

# In[4]:


# Print the first 5 rows of the train dataframe.
train_df.head()


# In[5]:


# Print the first 5 rows of the train dataframe.
# note their is no Survived column here which is our target varible we are trying to predict
test_df.head()


# In[6]:


# lets print data info
train_df.info()


# _**From the above information we can say that the training-set has a total of 891 examples and 11 features + the target variable (survived). 2 of the features are floats (Age, Fare), 5 are integers(PassengerId, Survived, Pclass, SibSp, Parch) and 5 are objects(Name, Sex, Ticket, Cabin, Embarked). Below I have listed the features with a short description:**_
# 
# 
#     survival:	Survival
#     PassengerId: Unique Id of a passenger.
#     pclass:	Ticket class	
#     sex:	Sex	
#     Age:	Age in years	
#     sibsp:	# of siblings / spouses aboard the Titanic	
#     parch:	# of parents / children aboard the Titanic	
#     ticket:	Ticket number	
#     fare:	Passenger fare	
#     cabin:	Cabin number	
#     embarked:	Port of Embarkation
# 

# In[7]:


train_df.describe()


# Above we can see that **38% out of the training-set survived the Titanic**. We can see that Age and Fare are measured on very different scaling, So we need to do feature scaling before predictions. On top of that we can already detect some features, that contain missing values, like the **'Age'** feature.

# ## **Visualizing Data**
# 
# *Lets compare certain features with survival to see which one **correlates** better. Also Visualizing data is crucial for recognizing underlying patterns to exploit in the model.*

# In[8]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_df);
# From below we can see that Male/Female that Emarked from C have higher chances of survival compared to other Embarked points.


# In[9]:


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);


# In[10]:


g = sns.FacetGrid(train_df, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple");


# In[11]:


corr=train_df.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# ## **Dealing with missing values**
# 
# **Its important to fill missing values, because some machine learning algorithms can't accept them eg SVM.**
# 
# *On the contrary filling missing values with the mean/median/mode is also a prediction which may not be 100% accurate, instead you can use models like Decision Trees and Random Forest which handles missing values very well.*

# In[12]:


#lets see which are the columns with missing values in train dataset
train_df.isnull().sum()


# In[13]:


labels = []
values = []
null_columns = train_df.columns[train_df.isnull().any()]
for col in null_columns:
    labels.append(col)
    values.append(train_df[col].isnull().sum())

ind = np.arange(len(labels))
width=0.6
fig, ax = plt.subplots(figsize=(6,5))
rects = ax.barh(ind, np.array(values), color='purple')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values in train dataset");


# In[14]:


#lets see which are the columns with missing values in test dataset
test_df.isnull().sum()


# In[15]:


labels = []
values = []
null_columns = test_df.columns[test_df.isnull().any()]
for col in null_columns:
    labels.append(col)
    values.append(test_df[col].isnull().sum())

ind = np.arange(len(labels))
width=0.6
fig, ax = plt.subplots(figsize=(6,5))
rects = ax.barh(ind, np.array(values), color='purple')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values in test dataset");


# ### Fill Missing Values in Embarked Column (train dataset)

# In[16]:


#Lets check which rows have null Embarked column
train_df[train_df['Embarked'].isnull()]


# **PassengerId 62 and 830** have missing embarked values
# 
# Both have ***Passenger class 1*** and ***fare $80.***
# 
# Lets plot a graph to visualize and try to guess from where they embarked

# In[17]:


from numpy import median
sns.barplot(x="Embarked", y="Fare", hue="Pclass", data=train_df, estimator=median)


# We can see that for ***1st class*** median line is coming around ***fare $80*** for ***embarked*** value ***'C'***.
# So we can replace NA values in Embarked column with 'C'

# In[18]:


train_df["Embarked"] = train_df["Embarked"].fillna('C')


# ### Fill Missing Values in Fare Column (test dataset)

# In[19]:


#Lets check which rows have null Fare column in test dataset
test_df[test_df['Fare'].isnull()]


# In[20]:


# we can replace missing value in fare by taking median of all fares of those passengers
# who share 3rd Passenger class and Embarked from 'S' , so lets find out those rows
test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')].head()


# In[21]:


# now lets find the median of fare for those passengers.

def fill_missing_fare(df):
    median_fare = df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median() #'S'
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

test_df = fill_missing_fare(test_df)


# ### Fill Missing Values in Age Column (train & test dataset)
# 
# **When dealing with large missing values in a particular column, simply removing the feature does not make sense when the feature is relevant in prediction.**
# 
# Age seems to be promising feature.
# So it doesnt make sense to simply fill null values out with median/mean/mode.
# 
# We will use ***Random Forest*** algorithm to predict ages. 

# In[22]:


# age distribution in train dataset
with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(train_df["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="red")
    plt.title("Age Distribution")
    plt.ylabel("Count")


# In[23]:


# age distribution in test dataset
with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(test_df["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="red")
    plt.title("Age Distribution")
    plt.ylabel("Count")


# In[24]:


# predicting missing values in age using Random Forest
# import the RandomForestRegressor Object
from sklearn.ensemble import RandomForestRegressor

def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Pclass','SibSp','Parch','Fare']]
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


# In[25]:


train_df = fill_missing_age(train_df)
test_df = fill_missing_age(test_df)


# _**Note: We will deal some of the features with missing values in Feature Engineering Section**_

# ## **Feature Engineering**

# ### Cabin
# **We will build a new feature called 'Deck' from Cabin, and remove the Cabin feature from the final dataset**

# In[26]:


train_df["Deck"] = train_df.Cabin.str[0] # the first character denotes the deck the passenger is allocated
test_df["Deck"] = test_df.Cabin.str[0]
train_df["Deck"].unique() # 0 is for null values


# In[27]:


sns.set(font_scale=1)
g = sns.factorplot("Survived", col="Deck", col_wrap=4,
                    data=train_df[train_df.Deck.notnull()],
                    kind="count", size=2.5, aspect=.8);


# _From above figure we can see that Deck C & B has higher number of survival, contrary they have higher number of non-survival too._

# In[28]:


train_df.Deck.fillna('Z', inplace=True)
test_df.Deck.fillna('Z', inplace=True)
# train_df = train_df.assign(Deck=train_df.Deck.astype(object)).sort_values("Deck")
sorted_deck_values = train_df["Deck"].unique() # Z is for null values
sorted_deck_values.sort()
sorted_deck_values


# In[29]:


# Drop Cabin Feature from final dataset for test and train
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# ### **SibSp and Parch**:
# SibSp and Parch would make more sense as a combined feature, that shows the total number of relatives, a person has on the Titanic. I will create it below and also a feature that shows if someone is not alone.

# In[30]:


data = [train_df, test_df]

for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

train_df['not_alone'].value_counts()


# In[31]:


sns.set(font_scale=1)
sns.factorplot("Survived", col="relatives", col_wrap=4,
                    data=train_df,
                    kind="count", size=2.5, aspect=.8);


# In[32]:


sns.catplot('relatives','Survived',kind='point', 
                      data=train_df, aspect = 2.5, )


# Here we can see that you had a high probabilty of survival with 1 to 3 realitves, but a lower one if you had less than 1 or more than 3 (except for some cases with 6 relatives).

# ### Name
# **We will build a new feature called 'Title' from Name, and remove the Name feature from the final dataset**

# In[33]:


data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[34]:


# Drop Name Feature from final dataset for test and train
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# ## **Converting Features , Transform the Data and Cleanup**
# In below section we will
# - convert categorical features(Title,) into numerical type.
# - convert sex feature into numeric
# - convert deck feature into numeric
# - convert embarked feature intor numeric
# - Feature Scaling 'Age' & 'Fare' 
# - drop ticket feature

# In[35]:


train_df.head()


# ### **Converting Title Feature**

# In[36]:


data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    
    # filling NaN with 0, to be safe
    dataset['Title'] = dataset['Title'].fillna(0)


# ### **Converting 'Sex' feature intor numeric**

# In[37]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# ### **Converting 'Deck' feature intor numeric**

# In[38]:


deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Z": 9}
data = [train_df, test_df]

for dataset in data:
    dataset['Deck'] = dataset['Deck'].map(deck)


# ### **Converting 'Embarked' feature intor numeric**

# In[39]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# ### **Drop Ticket Feature**

# In[40]:


train_df['Ticket'].describe()


# Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.

# In[41]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# ## **Feature Scaling**
# We can see that Age, Fare are measured on different scales, so we need to do Feature Scaling first before we proceed with predictions.

# In[42]:


from sklearn import preprocessing

data = [train_df, test_df]

for dataset in data:
    std_scale = preprocessing.StandardScaler().fit(dataset[['Age', 'Fare']])
    dataset[['Age', 'Fare']] = std_scale.transform(dataset[['Age', 'Fare']])


# In[43]:


train_df.head()


# In[44]:


test_df.head()


# ## **Start with Prediction**
# 
# ### **Splitting up the training & test data.**
# 
# In this section we will be dealing with train data, we will be splitting the train **(train_df)** data into 80 % train and 20% test data and will evaluate the model performance on the final test data **(test_df)**
# 
# First, separate the features(X) from the labels(y). 
# 
# **X_all:** All features minus the value we want to predict (Survived).
# 
# **y_all:** Only the value we want to predict. 
# 
# Second, we use Scikit-learn to randomly shuffle this data into four variables. In this case I will be splitting the train data **(train_df)** into 80 % train and 20% test data  
# 
# Later, this data will be reorganized into a KFold pattern to train on various algorithms. 
# 
# And finally evaluate the effectiveness of trained algorithms on the final test data **(test_df)**.

# In[45]:


# from sklearn.model_selection import train_test_split

# X_all = train_df.drop(['Survived', 'PassengerId'], axis=1)
# y_all = train_df['Survived']

# # define seed for reproducibility
# seed = 1

# num_test = 0.20
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=seed)


# In[46]:


from sklearn.model_selection import train_test_split

# # define seed for reproducibility
seed = 1

X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
y_test  = pd.read_csv("../input/gender_submission.csv")['Survived']


# In[47]:


len(y_test)


# ### Training and Testing the Classification Algorithms
# 
# Now that we have preprocessed the data and built our training and testing datasets, we can start to deploy different classification algorithms. It's relatively easy to test multiple models; as a result, we will compare and contrast the performance of 8 different algorithms.

# In[48]:


# Now that we have our dataset, we can start building algorithms! 
# We'll need to import each algorithm we plan on using

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # (Naive Bayes)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# define scoring method
scoring = 'accuracy'

# Define models to train

names = ["Random Forest", "AdaBoost","Nearest Neighbors", 
         "Naive Bayes","Decision Tree","Logistic Regression","Gaussian Process",
         "SVM RBF", "SVM Linear", "SVM Sigmoid"]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    KNeighborsClassifier(n_neighbors = 3),
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression(random_state = 0),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    SVC(kernel = 'rbf', random_state=0),
    SVC(kernel = 'linear', random_state=0),
    SVC(kernel = 'sigmoid', random_state=0)
]

models = dict(zip(names, classifiers))

# models = {
#     "Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     "AdaBoost":AdaBoostClassifier(),
#     "Nearest Neighbors":KNeighborsClassifier(n_neighbors = 3),
#     "Naive Bayes":GaussianNB(),
#     "Decision Tree":DecisionTreeClassifier(max_depth=5),
#     "Logistic Regression":LogisticRegression(random_state = 0),
#     "Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0)),
#     "SVM RBF":SVC(kernel = 'rbf'),
#     "SVM Linear":SVC(kernel = 'linear'),
#     "SVM Sigmoid":SVC(kernel = 'sigmoid')
# }


# In[49]:


# Remember, performance on the training data is not that important. We want to know how well our algorithms
# can generalize to new data.  To test this, let's make predictions on the validation dataset.

models_accuracy_score = []

for name in models:
    model = models[name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    models_accuracy_score.append((name, accuracy_score(y_test, predictions)))
    print("Model name: ", name)
    print("Model accuracy score: ", accuracy_score(y_test, predictions))


# ### **Which is the best Model ?**

# In[50]:


results = pd.DataFrame({
    'Model': names,
    'Score': [curr_model_score[1] for curr_model_score in models_accuracy_score]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# As we can see that SVM Linear classifier goes on the first place. But first, let us check, how svm linear classifier performs, when we use k-fold cross validation.

# ### **K-Fold Cross Validation**

# In[51]:


# import K-fold class
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state = seed)
model = models["SVM Linear"]
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

print("{}: {} ({})".format("SVM Linear", cv_results.mean(), cv_results.std()))


# ## **Hyperparameter Tuning**

# In[ ]:


# Applying Grid Search to find the best model and the best parameters,
# i.e. we will find whether our problem is linear or non-linear

from sklearn.model_selection import GridSearchCV

classifier = models["SVM Linear"]

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear'], 
                'gamma': [0.1, 0.01, 0.001, 0.0001]}]

# parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 
#                'gamma': [0.1, 0.01, 0.001, 0.0001]},
#                 {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 
#                'gamma': [0.1, 0.01, 0.001, 0.0001]}
#              ]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


print(best_accuracy)
print(best_parameters)


# In[ ]:


# perform classification again using optimal parameter from grid search cv

svm_linear_classifier = SVC(C=10, kernel='rbf', gamma=0.01)
svm_linear_classifier.fit(X_train, y_train)
predictions = svm_linear_classifier.predict(X_test)

print("Model accuracy score: ", accuracy_score(y_test, predictions))
print("Model score: ", svm_linear_classifier.score(X_train, y_train) , "%")


# In[ ]:


# perform classification again using optimal parameter from grid search cv

svm_linear_classifier = SVC(C=10, kernel='rbf', gamma=0.01)
svm_linear_classifier.fit(X_train, y_train)
predictions = svm_linear_classifier.predict(X_test)

print("Model accuracy score: ", accuracy_score(y_test, predictions))
print("Model score: ", svm_linear_classifier.score(X_train, y_train) , "%")


# In[ ]:


# confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# predictions = cross_val_predict(svm_linear_classifier, X_train, y_train, cv=3)
confusion_matrix(y_test, predictions)

