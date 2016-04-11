import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


#### converting variables and clean the data
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train["Embarked"] = train["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

train["Age"] = train["Age"].fillna(train["Age"].median())

## building the first tree
target = np.array(train.Survived).transpose()
features_one = np.array([train.Pclass, train.Sex, train.Age,  train.Fare]).transpose()

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

#### second tree

features_two = np.array([train.Pclass,train.Age,train.Sex, train.Fare, train.SibSp, train.Parch,train.Embarked]).transpose()

my_tree_two = tree.DecisionTreeClassifier()
my_tree_two = my_tree_two.fit(features_two, target)

#### third tree
# control overfitting
my_tree_three = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)
my_tree_three = my_tree_three.fit(features_two, target)


### evaluating the models
from sklearn.metrics import confusion_matrix

pred_vec_three = my_tree_three.predict(features_two)
pred_vec_two = my_tree_two.predict(features_two)
pred_vec_one = my_tree_one.predict(features_one)

def pred_eval(pred_vec,target):
    cm = confusion_matrix(pred_vec,target)
    true_positive = cm[0][0]
    true_negative = cm[1][1]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    positive = true_positive + false_negative
    negative = true_negative + false_positive
    sensitivity = true_positive/positive #proportion of survivals correctly classified (want to maximize)
    specificity = true_negative/negative #proportion of deaths correctly classified (want to maximize)
    ppv = true_positive/(true_positive + false_positive)
    npv = true_negative/(true_negative + false_negative)
    fnr = false_negative/positive #accordingly minimize 1 - sensitivity
    fpr = false_positive/negative #accordingly minimize 1 - specificity
    
    eval = np.array([cm,sensitivity,specificity,ppv,npv,fnr,fpr])
    return(eval)

my_tree_one.score(features_one, target)
my_tree_two.score(features_two, target)
my_tree_three.score(features_two, target)

#### Graphiong the Tree


#from sklearn.externals.six import StringIO 
#import pydot
#dot_data = StringIO() 
#tree.export_graphviz(my_tree_one, out_file = dot_data)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("tree.pdf")

#from sklearn.externals.six import StringIO
#with open("tree.dot", 'w') as f:
#    f = tree.export_graphviz(my_tree_two, out_file=f)

#from IPython.display import Image
#dot_data = StringIO()
#tree.export_graphviz(my_tree_two, out_file=dot_data,  filled=True, rounded=True,  special_characters=True)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())



#### Useful Attributes
my_tree_one.feature_importances_
my_tree_one.tree_
my_tree_one.n_classes_
my_tree_one.n_features_
my_tree_one.classes_



####  Clean the test data.
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test["Embarked"] = test["Embarked"].fillna("S")

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())

test.Fare[152] = test.Fare.median()


#### Prediction

test_features_one = np.array([test.Pclass, test.Fare, test.SibSp, test.Parch]).transpose()
pred_one = my_tree_one.predict(test_features_one)


test_features_two = np.array([test.Pclass,test.Age,test.Sex, test.Fare, test.SibSp, test.Parch,test.Embarked]).transpose()
pred_two = my_tree_two.predict(test_features_two)

pred_three = my_tree_three.predict(test_features_two)


#### Feature Engineering


#### https://plot.ly/matplotlib/bar-charts/

y1 = cm1[1:5] 
y2 = cm2[1:5]
y3 = cm3[1:5]
N = len(y1)
x = range(N)
plt.bar(x, y2, color="red")
plt.bar(x, y3, color="green")
plt.bar(x, y1, color="blue")

g1 = cm1[5:7] 
g2 = cm2[5:7]
g3 = cm3[5:7]
M = len(g1)
h = range(M)
plt.bar(h, g1, color="blue")
plt.bar(h, g3, color="green")
plt.bar(h, g2, color="red")


#### Building a Random Forest

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

features_forest = np.array([train.Pclass,train.Age,train.Sex, train.Fare, train.SibSp, train.Parch,train.Embarked]).transpose()

forest = RandomForestClassifier(max_depth = 10, n_estimators=100, min_samples_split=2)
my_forest = forest.fit(features_forest, target)
my_forest.score(features_forest, target)

#Evaluate the forest
pred_vec_forest = my_forest.predict(features_forest)
pred_eval(pred_vec_forest,target)

#predict using the forest
pred_forest = my_forest.predict(test_features_two)



