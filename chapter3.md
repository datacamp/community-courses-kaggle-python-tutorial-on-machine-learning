---
title       : Improving your predictions through Random Forests 
description : "What techniques can you use to improve your predictions even more? One possible way is by making use of the machine learning method Random Forest. Namely, a forest is just a collection of trees..."
attachments :


--- type:NormalExercise lang:python xp:100 skills:2
## A Random Forest analysis in Python
A detailed study of Random Forests would take this tutorial a bit too far. However, since it's an often used machine learning technique, gaining a general understanding in Python won't hurt.

In layman's terms, the Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

Building a random forest in Python looks almost the same as building a decision tree; so we can jump right to it. There are two key differences, however. Firstly, a different class is used. And second, a new argument is necessary. Also, we need to import the necessary library from scikit-learn.

- Use `RandomForestClassifier()` class instead of the `DecisionTreeClassifier()` class. 
- `n_estimators` needs to be set when using the `RandomForestClassifier()` class. This argument allows you to set the number of trees you wish to plant and average over.

The latest training and testing data are preloaded for you.


*** =instructions
- Import `RandomForestClassifier` from `from sklearn.ensemble`.
- Build an array with features we used for the most recent tree and call it features_forest.
- Build the random forest with `n_estimators` set to `100`.
- Build an array with the features from the test set to make predictions. Use this array and the model to compute the predictions.


*** =hint

- When computing the predictions you can use the `.predict()` mothod just like you did with decision trees!
- To Compute the score use the `.score()` method with correct argumnets. Consult your previous work from CH2 if your don't recall the syntax.

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"] = train["Embarked"].fillna("S")

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"].values

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)


test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"] = test["Embarked"].fillna("S")

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())

test.Fare[152] = test.Fare.median()

```

*** =sample_code
```{python}
#Import the `RandomForestClassifier`
from sklearn.ensemble import ___

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = 

#Building the Forest: my_forest
n_estimators = 
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, ___, random_state = 1)
my_forest = forest.fit(features_forest, target)

#Print the score of the random forest


#Compute predictions and print the length of the prediction vector:test_features, pred_forest
test_features = 
pred_forest = 
print()
```

*** =solution
```{python}

#Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

#We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

#Building the Forest: my_forest
n_estimators = 100
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = n_estimators, random_state = 1)
my_forest = forest.fit(features_forest, target)

#Print the score of the random forest
print(my_forest.score(features_forest, target))

#Compute predictions and print the length of the prediction vector:test_features, pred_forest
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

```

*** =sct

```{python}
test_function("RandomForestClassifier", args=None,
              incorrect_msg = "Don't forget to import `RandomForestClassifier` and use it to initiate your random forest.")
test_object("n_estimators",
              incorrect_msg = "We are looking to generate 100 estimators. Make sure to set this argument correctly!")
test_object("features_forest",
              incorrect_msg = "Make sure to select the specified features in the right order. These should come from the train dataset!")
test_function("print",1, args=None,
              incorrect_msg = "It looks like the score wasn't computet exactly right. Make sure to use `features_forest` and `target` as arguments")
test_object("test_features",
            incorrect_msg = "Make sure to select the specified features in the right order. These should come from the test dataset!")
test_function("print",2, args=None,
            incorrect_msg = "It seems that there is an incorrect number of predictions is pred_forest. Make sure to use `test_features` when computing the predictions.")
```

--- type:NormalExercise lang:python xp:100 skills:2
## Interpreting and Comparing

Remember how we looked at `.feature_importances_` attribute for the decision trees? Well, you can request the same attribute from your random forest as well and interpret the relevance of the included variables.
You might also want to compare the models in some quick and easy way. For this, we can use the `.score()` method. The `.score()` method takes the features data and the target vector and computes mean accuracy of your model. You can apply this method to both the forest and individual trees. Remember, this measure should be high but not extreme because that would be a sign of overfitting.

For this exercise, you have `my_forest` and `my_tree_two` available to you. The features and target arrays are also ready for use.

*** =instructions
- Explore the feature importance for both models
- Compare the mean accuracy score of the two models

*** =hint

- Make sure that you are applying the commands to `my_forest` and, are using correct arguments.
- Don't forget that `target` and `features_forest` are preloaded for you!

*** =pre_exercise_code
```{python}
import random
random.seed(1)

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"].values

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=100, random_state = 1)
my_forest = forest.fit(features_forest, target)

```


*** =sample_code
```{python}
#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print()

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print()
```

*** =solution
```{python}
#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))
```
*** =sct

```{python}

test_function("print", 1, args=None,
              incorrect_msg = "You don't need to edit the given code. Instead leave it and use it as a hint for your solution")
test_function("print", 2, args=None,
              incorrect_msg = "Use the give code as a hint on how to complete the task. You solution shoudl look the same except with `my_forest` and an object of investigation!")
test_function("print", 3, args=None,
              incorrect_msg = "You don't need to edit the given code. Instead leave it and use it as a hint for your solution")
test_function("print", 4, args=None,
              incorrect_msg = "Use the give code as a hint on how to complete the task. You solution shoudl look the same except with `my_forest` and an object of investigation!")

```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Conclude and Submit

Based on your finding in the previous exercise determine which feature was of most importance, and for which model.
After this final exercise, you will be able to submit your random forest model to Kaggle! Use `my_forest`, `my_tree_two`, and `feature_importances_` to answer the question.

*** =hint

- By significance, we simply mean the magnitude of the values. For each feature you should see a decimal. The largst indicates greatest significance for the respective feature.

*** =pre_exercise_code

```{python}

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"].values

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)
my_tree_two = my_tree_two.fit(features_two, target)

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=100)
my_forest = forest.fit(features_forest, target)

```

*** =instructions
- `The most important feature was "Age", but it was more significant for "my_tree_two"`
- `The most important feature was "Sex", but it was more significant for "my_tree_two"`
- `The most important feature was "Sex", but it was more significant for "my_forest"`
- `The most important feature was "Age", but it was more significant for "my_forest"`

*** =sct

```{python}

msg1 = "Wrong choice. Check the hint for some help."
msg2 = "Wonderful! You are now at the end of this tutorial and ready to start improving the results yourself"
msg3 = msg1
msg4 = msg1
test_mc(correct = 2, msgs = [msg1, msg2, msg3, msg4])

success_msg("Congrats on compleating the course! Now that you created your first random forest and used it for prediction take a look at how well it does in the Kaggle competition. [Download your csv file](https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/my_solution_forest.csv). Having learned about decision trees and random forests, you can begin participating in some other Kaggle competitons as well. Good luck and have fun!")

```
