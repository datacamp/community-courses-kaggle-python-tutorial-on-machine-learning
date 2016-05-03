---
title       : Predicting with Decision Trees
description : After making your first predictions in the previous chapter, it's time to bring you to the next level. In chapter 2 you
will be introduced to a fundamental concept in machine learning: decision trees.
attachments :


--- type:NormalExercise lang:python xp:100 skills:2 key:98be5c3225
## Intro to decision trees

In the previous chapter, you did all the slicing and dicing yourself to find subsets that have a higher chance of surviving. A decision tree automates this process for you and outputs a classification model or classifier.

Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat. The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations in that node determine how to predict for new observations that end up in that terminal node.

First, let's import the necessary libraries:

*** =instructions
- Import the `numpy` library as `np`
- From `sklearn` import the `tree`


*** =hint

- Use the `import` and `as` special keys when importing `numpy`.
- You can use `from sklearn import tree` command to import `tree`.

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

```

*** =sample_code
```{python}
#Import the Numpy library

#Import 'tree' from scikit-learn library
from sklearn 

```

*** =solution
```{python}
#Import the Numpy library
import numpy as np

#Import 'tree' from scikit-learn library
from sklearn import tree

```

*** =sct

```{python}

test_import("numpy", same_as = False)
success_msg("OK, your package is loaded now. Time for the real deal.")

```


--- type:NormalExercise lang:python xp:100 skills:2 key:98092838ce
## Cleaning and Formatting your Data

Before you can begin constructing your trees you need to get your hands dirty and clean the data so that you can use all the features available to you. In the first chapter, we saw that the Age variable had some missing value. Missingness is a whole subject with and in itself, but we will use a simple imputation technique where we substitute each missing value with the median of the all present values.

```
train["Age"] = train["Age"].fillna(train["Age"].median())
```

Another problem is that the Sex and Embarked variables are categorical but in a non-numeric format. Thus, we will need to assign each class a unique integer so that Python can handle the information. Embarked also has some missing values which you should impute witht the most common class of embarkation, which is `"S"`.


*** =instructions
- Assign the integer 1 to all females
- Impute missing values in `Embarked` with class `S`. Use `.fillna()` method.
- Replace each class of Embarked with a uniques integer. `0` for `S`, `1` for `C`, and `2` for `Q`.
- Print the `Sex` and `Embarked` columns

*** =hint
- Use the standard bracket notation to select the appropriate rows and columns, and don't foget the `==` operator.

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

```

*** =sample_code
```{python}
#Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0

#Impute the Embarked variable
train["Embarked"] = 

#Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0

#Print the Sex and Embarked columns

```

*** =solution
```{python}
#Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train["Sex"])
print(train["Embarked"])
```
*** =sct

```{python}

msg = "It looks like you coded the `Sex` variable incorecctly. Make sure to use `0` for male and `1` for female"
test_function("print", 1,
              args=None,
              not_called_msg = msg,
              incorrect_msg = msg,)

msg = "It looks like you coded the `Embarked` variable incorecctly. Make sure to use `0` for `S`, `1` for `C, and `2` for `Q`."
test_function("print", 2,
              args=None,
              not_called_msg = msg,
              incorrect_msg = msg,)

success_msg("Geat! Now that the data is cleaned up a bit you are ready to begin building your first decision tree.")
```

--- type:NormalExercise lang:python xp:100 skills:2 key:2b663996b1
## Creating your first decision tree

You will use the `scikit-learn` and `numpy` libraries to build your first decision tree. `scikit-learn` can be used to create `tree` objects from the `DecisionTreeClassifier` class. The methods that we will use take `numpy` arrays as inputs and therefore we will need to create those from the `DataFrame` that we already have. We will need the following to build a decision tree

- `target`: A one-dimensional numpy array containing the target/response from the train data. (Survival in your case)
- `features`: A multidimensional numpy array containing the features/predictors from the train data. (ex. Sex, Age)

Take a look at the sample code below to see how this would look like:

```
target = train["Survived"].values

features = train[["Sex", "Age"]].values

my_tree = tree.DecisionTreeClassifier()

my_tree = my_tree.fit(features, target)

```

One way to quickly see the result of your decision tree is to see the importance of the features that are included. This is done by requesting the `.feature_importances_` attribute of your tree object. Another quick metric is the mean accuracy that you can compute using the `.score()` function with `features` and `target` as arguments.

Ok, time for you to build your first decision tree in Python! The train and testing data from chapter 1 are available in your workspace.

*** =instructions
- Build the `target` and `features` numpy arrays. The target will be based on the Survived column in `train`. The features
array will be based on the variables Passenger Class, Sex, Age, and Passenger Fare
- Build a decision tree `my_tree_one` to predict survival using `features` and `target`
- Look at the importance of features in your tree and compute the score

*** =hint
- To build a tree use the `tree.DecisionTreeClassifier()` syntax.
- You can look at the importance of features using the `.feature_importances_` attribute.



*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

```

*** =sample_code
```{python}
#Print the the train data to see the available features
print(train)


#Create the target and features numpy arrays: target, features_one



#Fit your first decision tree: my_tree_one



#Look at the importance of the included features and print the score

```

*** =solution

```{python}
#Print the train data to see the available features
print(train)

#Create the target and features numpy arrays: target, features

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

#Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

#Look at the importance of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

```


*** =sct

```{python}
msg = "`target` should be the `Survived` variable from the train dataset. Follow the code in the discussion for help."
test_object("target",
              undefined_msg = msg,
              incorrect_msg = msg)

msg = "Make sure that you are including the correct features in the stated order. Follow the code in the discussion for help."
test_object("features_one",
              undefined_msg = msg,
              incorrect_msg = msg)

msg = "It looks like the score was not computed correctly. Try re-submitting the code!"
test_function("print",3,
              args=None,
              not_called_msg =msg,
              incorrect_msg = msg)

success_msg("Well done! Time to investigate your decision tree a bit more.")
```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:87b643ee96
## Interpreting your decision tree

The `feature_importances_` attribute make it simple to interpret the significance of the predictors you include. Based on your decision tree, what variable plays the most important role in determining whether or not a passenger survived? Your model (`my_tree_one`) is available in the console.

*** =instructions
- Passenger Class
- Sex/Gender
- Passenger Fare
- Age

*** =hint
Have a close look at the `feature_importances_` attribute of your tree. What variable has the greatest coefficient? 

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier(random_state = 1)
my_tree_one = my_tree_one.fit(features_one, target)


```

*** =sct

```{python}
msg1 = "Wrong choice. Check the hint for some help."
msg3 = "Bellissimo! Time to make a prediction and submit it to Kaggle!"
msg2 = msg1
msg4 = msg1
test_mc(correct = 3, msgs = [msg1, msg2, msg3, msg4])

success_msg("Looks like Passenger Fare has most significance in determining survival based on your model. Now let's move on to making your first submission to Kaggle!")

```


--- type:NormalExercise lang:python xp:100 skills:2 key:4a70446ddd
## Predict and submit to Kaggle

To send a submission to Kaggle you need to predict the survival rates for the observations in the test set. In the last exercise of the previous chapter, we created simple predictions based on a single subset. Luckily, with our decision tree, we can make use of some simple functions to "generate" our answer without having to manually perform subsetting.

First, you make use of the `.predict()` method. You provide it the model (`my_tree_one`), the values of features from the dataset for which predictions need to be made (`test`). To extract the features we will need to create a numpy array in the same way as we did when training the model. However, we need to take care of a small but important problem first. There is a missing value in the Fare feature that needs to be imputed.

Next, you need to make sure your output is in line with the submission requirements of Kaggle: a csv file with exactly 418 entries and two columns: `PassengerId` and `Survived`. Then use the code provided to make a new data frame using `DataFrame()`, and create a csv file using `to_csv()` method from Pandas. 

*** =instructions
- Impute the missing value for Fare in row 153 with the median of the column.
- Make a prediction on the test set using the `.predict()` method and `my_tree_one`. Assign the result to `my_prediction`.
- Create a data frame `my_solution` containing the solution and the passenger ids from the test set. Make sure the solution is in line with the standards set forth by Kaggle by naming the column appropriately.

*** =hint

- When doing the imputation use the `Fare` feature and the `.median` method.
- Make sure to select the Pclass, Sex, Age, and Fare features in this exact order. Don't chnage the skeleton of the solution!


*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

target = train["Survived"].values

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier(random_state = 1)
my_tree_one = my_tree_one.fit(features_one, target)

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

```

*** =sample_code

```{python}
# Impute the missing value with the median
test.Fare[152] = 

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[[___, ___, ___, ___]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
```

*** =solution

```{python}
# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
```

*** =sct

```{python}

test_object("test_features",
            incorrect_msg = "Make sure that you are selecting the correct variables from the `test` dataset.")
test_function("print",3, args=None,
            incorrect_msg = "It looks like your solution doesn't have the correct number of entries. There should be exactly 418 rows!")

success_msg("Great! You just created your first decision tree. [Download your csv file](https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/my_solution_one.csv), and submit the created csv to Kaggle to see the result of your effort.")

```

--- type:NormalExercise lang:python xp:100 skills:2 key:fa5a95aab5
## Overfitting and how to control it.

When you created your first decision tree the default arguments for  `max_depth` and `min_samples_split` were set to `None`. This means that no limit on the depth of your tree was set.  That's a good thing right? Not so fast. We are likely overfitting. This means that while your model describes the training data extremely well, it doesn't generalize to new data, which is frankly the point of prediction. Just look at the Kaggle submission results for the simple model based on Gender and the complex decision tree. Which one does better?

Maybe we can improve the overfit model by making a less complex model? In `DecisionTreeRegressor`, the depth of our model is defined by two parameters:
- the `max_depth` parameter determines when the splitting up of the decision tree stops.
- the `min_samples_split` parameter monitors the amount of observations in a bucket. If a certain threshold is not reached (e.g minimum 10 passengers) no further splitting can be done.

By limiting the complexity of your decision tree you will increase its generality and thus its usefulness for prediction!
*** =instructions
- Include the Siblings/Spouses Aboard, Parents/Children Aboard, and Embarked features in a new set of features.
- Fit your second tree `my_tree_two` with the new features, and control for the model compelexity by toggling the `max_depth` and `min_samples_split` arguments.


*** =hint

You can always use `train.describe()` in the console to check the names of the features.

*** =pre_exercise_code

```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

target = train["Survived"].values

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

```

*** =sample_code
```{python}
# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", ___, ___, ___]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 
min_samples_split =
my_tree_two = tree.DecisionTreeClassifier(max_depth = ___, min_samples_split = ____, random_state = 1)
my_tree_two = 

#Print the score of the new decison tree

```


*** =solution
```{python}
# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))
```

*** =sct

```{python}
test_object("features_two",
            incorrect_msg = "Make sure you are selecting the specified features from the train dataset.")
test_object("max_depth",
            incorrect_msg = "The maximum deapth argument shoudl be set to 10!")
test_object("min_samples_split",
            incorrect_msg = "The min_samples_split argument shoudl be set to 5!")
test_function("print", args=None,
            incorrect_msg = "It looks like score wasn't computed quite right. Make sure that the you are using the `features_two` and `target` as your arguments.")

success_msg("Great! You just created your second and possibly improved decision tree. [Download your csv file](https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/my_solution_two.csv) .Submit your updated solution to Kaggle to see how despite a lower `.score` you predict better.")

```

--- type:NormalExercise lang:python xp:100 skills:2 key:55678ebefb
## Feature-engineering for our Titanic data set

Data Science is an art that benefits from a human element. Enter feature engineering: creatively engineering your own features by combining the different existing variables. 

While feature engineering is a discipline in itself, too broad to be covered here in detail, you will have a look at a simple example by creating your own new predictive attribute: `family_size`.  

A valid assumption is that larger families need more time to get together on a sinking ship, and hence have lower probability of surviving. Family size is determined by the variables `SibSp` and `Parch`, which indicate the number of family members a certain passenger is traveling with. So when doing feature engineering, you add a new variable `family_size`, which is the sum of `SibSp` and `Parch` plus one (the observation itself), to the test and train set.

*** =instructions
- Create a new train set `train_two` that differs from `train` only by having an extra column with your feature engineered variable `family_size`.
-  Add your feature engineered variable `family_size` in addition to `Pclass`, `Sex`, `Age`, `Fare`, `SibSp` and `Parch` to `features_three`.
- Create a new decision tree as `my_tree_three` and fit the decision tree with your new feature set `features_three`. Then check out the score of the decision tree.

*** =hint

- Don't forget to add `1` when adding the column with the new feature 
- Add your newly defined feature to be included in `features_three`
- Remember how you fit the decision tree model in the last exercise

*** =pre_exercise_code
```{python}
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

target = train["Survived"].values

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

```

*** =sample_code
```{python}
# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = 

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", ___]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = 

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))

```

*** =solution

```{python}
# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train["SibSp"] + train["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))

```

*** =sct

```{python}

test_object("features_three",
            incorrect_msg = "Be sure that you add 1 while defining `family_size`. Then add `family_size` to `features_three`.")
test_function("print", args=None,
            incorrect_msg = "It looks like score wasn't computed quite right. Make sure that the you are using the `features_three` and `target` to fit your decision tree model.")

success_msg("Great! Notice that this time the newly created variable is included in the model. [Download your csv file](https://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/my_solution_three.csv), and submit the created csv to Kaggle to see the result of the updated model.")
```

