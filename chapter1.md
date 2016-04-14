---
title       : Getting Started with Python
description : In this chapter we will go trough the essential steps that you will need to take before beginning to build predictive models.
attachments :

--- type:NormalExercise lang:python xp:100 skills:2

## How it works
Welcome to our Kaggle Machine Learning Tutorial. In this tutorial, you will explore how to tackle Kaggle Titanic competition using Python and Machine Learning. In case you're new to Python, it's recommended that you first take our free [Introduction to Python for Data Science Tutorial](https://www.datacamp.com/courses/intro-to-python-for-data-science). Furthermore, while not required, familiarity with machine learning techniques is a plus so you can get the maximum out of this tutorial.

In the editor on the right, you should type Python code to solve the exercises. When you hit the 'Submit Answer' button, every line of code is interpreted and executed by Python and you get a message whether or not your code was correct. The output of your Python code is shown in the console in the lower right corner. Python makes use of the `#` sign to add comments; these lines are not run as Python code, so they will not influence your result.

You can also execute Python commands straight in the console. This is a good way to experiment with Python code, as your submission is not checked for correctness.


*** =instructions
- In the editor to the right, you see some Python code and annotations. This is what a typical exercise will look like.
- To complete the exercise and see how the interactive environment works  add the code to compute `y` and hit the `Submit Answer` button. Don't forget to print the result.


*** =hint

Just add a line of Python code that calculates the product of 6 and 9, just like the example in the sample code!

*** =pre_exercise_code
```{python}
# no pre_exercise_code
```

*** =sample_code
```{python}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
```

*** =solution
```{python}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
y = 6*9
print(y)
```

*** =sct
```{python}

msg = "Don't forget to assign the correct value to y"
test_object("y", 
            undefined_msg = msg, 
            incorrect_msg = msg)

msg = "Print out the resulting object, `y`!"
test_function("print",2, 
              not_called_msg = msg,
              incorrect_msg = msg,
              args=None)

success_msg("Awesome! See how the console shows the result of the Python code you submitted? Now that you're familiar with the interface, let's get down to business!")
```

--- type:NormalExercise lang:python xp:100 skills:2
## Get the Data with Pandas
When the Titanic sank, 1502 of the 2224 passengers and crew were killed. One of the main reasons for this high level of casualties was the lack of lifeboats on this self-proclaimed "unsinkable" ship.

Those that have seen the movie know that some individuals were more likely to survive the sinking (lucky Rose) than others (poor Jack). In this course, you will learn how to apply machine learning techniques to predict a passenger's chance of surviving using Python.

Let's start with loading in the training and testing set into your Python environment. You will use the training set to build your model, and the test set to validate it. The data is stored on the web as `csv` files; their URLs are already available as character strings in the sample code. You can load this data with the `read_csv()` method from the Pandas library.

*** =instructions
- First, import the Pandas library as pd.
- Load the test data similarly to how the train data is loaded.
- Print the first couple rows of the loaded dataframes using the `.head()` method.

*** =hint
- You can load in the training set with `train = pd.read_csv(train_url)`
- To print a variable to the console, use the print function on a new line.

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}
# Import the Pandas library

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

#Print the `head` of the train and test dataframes

```
*** =solution
```{python}
# Import the Pandas library
import pandas as pd

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())
```

*** =sct

```{python}
msg = "Have you correctly imported the `pandas` package? Use the alias `pd`."
test_import("pandas",  not_imported_msg = msg,  incorrect_as_msg = msg)

msg = "Do not touch the code that specifies the URLs of the training and test set csvs."
test_object("train_url", undefined_msg = msg, incorrect_msg = msg)
test_object("test_url", undefined_msg = msg, incorrect_msg = msg)

msg = "Make sure you are using the `read_csv()` function correctly"
test_function("pandas.read_csv", 1,
              args=None,
              not_called_msg = msg,
              incorrect_msg = msg,)
test_function("pandas.read_csv", 2,
              args=None,
              not_called_msg = msg,
              incorrect_msg = msg)

msg1 = "Don't forget to print the first few rows of the `train` with the `.head()` method"
msg2 = "Don't forget to print the first few rows of the `test` with the `.head()` method"
#test_function("print", 1, not_called_msg = msg1, incorrect_msg = msg1)

test_function("print", 2, not_called_msg = msg2, incorrect_msg = msg2)

success_msg("Well done! Now that your data is loaded in, let's see if you can understand it.")
```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2
## Understanding your data

Before starting with the actual analysis, it's important to understand the structure of your data. Both `test` and `train` are DataFrame objects, the way pandas represent datasets. You can easily explore a DataFrame using the `.describe()` method. `.describe()` summarizes the columns/features of the DataFrame, including the count of observations, mean, max and so on. Another useful trick is to look at the dimensions of the DataFrame. This is done by requesting the `.shape` attribute of your DataFrame object. (ex. `your_data.shape`)

The training and test set are already available in the workspace, as `train` and `test`. Apply `.describe()` method and print the `.shape` attribute of the training set. Which of the following statements is correct?

*** =instructions
- The training set has 891 observations and 12 variables, count for Age is 714.
- The training set has 418 observations and 11 variables, count for Age is 891.
- The testing set has 891 observations and 11 variables, count for Age is 891.
- The testing set has 418 observations and 12 variables, count for Age is 714.

*** =hint
To see the description of the `test` variable try `test.describe()`.

*** =pre_exercise_code
```{python}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sct

```{python}

msg1 = "Great job!"
msg2 = "Wrong, try again. Maybe have a look at the hint."
msg3 = "Not so good... Maybe have a look at the hint."
msg4 = "Incorrect. Maybe have a look at the hint."
test_mc(correct = 1, msgs = [msg1, msg2, msg3, msg4])

success_msg("Well done! Now move on and explore some of the features in more detail.")

```

--- type:NormalExercise lang:python xp:100 skills:1
## Rose vs Jack, or Female vs Male

How many people in your training set survived the disaster with the Titanic? To see this, you can use the `value_counts()` method in combination with standard bracket notation to select a single column of a DataFrame:

```
# absolute numbers
train["Survived"].value_counts()

# percentages
train["Survived"].value_counts(normalize = True)
``` 

If you run these commands in the console, you'll see that 549 individuals died (62%) and 342 survived (38%). A simple way to predict heuristically could be: "majority wins". This would mean that you will predict every unseen observation to not survive.

To dive in a little deeper we can perform similar counts and percentage calculations on subsets of the Survived column. For example, maybe gender could play a role as well? You can explore this using the `.value_counts()` method for a two-way comparison on the number of males and females that survived, with this syntax:

```
train["Survived"][train["Sex"] == 'male'].value_counts()
train["Survived"][train["Sex"] == 'female'].value_counts()
```

To get proportions, you can again pass in the argument `normalize = True` to the `.value_counts()` method.

*** =instructions
- Calculate and print the survival rates in absolute numbers using `values_counts()` method.
- Calculate and print the survival rates as proportions by setting the `normalize` argument to `True`.
- Repeat the same calculations but on subsets of survivals based on Sex.

*** =hint
- The code for the first four tasks is already given in the assignment!
- Think about the `normalize` argument, and don't forget to print.

*** =pre_exercise_code
```{python}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sample_code
```{python}

# Passengers that survived vs passengers that passed away
print()

# As proportions
print()

# Males that survived vs males that passed away
print()

# Females that survived vs Females that passed away
print()

# Normalized male survival
print()

# Normalized female survival
print()

```

*** =solution
```{python}

# Passengers that survived vs passengers that passed away
print(train.Survived.value_counts())

# As proportions
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))
```

*** =sct

```{python}
msg = "Make sure you are using `.value_counts()` method correctly."
test_function("print", 1,
              not_called_msg= msg,
              incorrect_msg = msg)

msg = "Don't forget to set `normalize = True` when using `.value_counts()`."
test_function("print", 2,
              not_called_msg = msg,
              incorrect_msg = msg)

msg = "Make sure you are partitioning by males."
test_function("print", 3,
              not_called_msg = msg,
              incorrect_msg = msg)

msg = "Make sure you are partitioning by females."
test_function("print", 4,
              not_called_msg= msg,
              incorrect_msg = msg)

msg = "Don't forget to set `normalize = True` when using `.value_counts()`."
test_function("print", 5,
              not_called_msg = msg,
              incorrect_msg = msg)

test_function("print", 6,
              not_called_msg = msg,
              incorrect_msg = msg)

success_msg("Well done! It looks like it makes sense to predict that all females will survive, and all men will die.")

```

--- type:NormalExercise lang:python xp:100 skills:2
## Does age play a role?

Another variable that could influence survival is age; it's probable that children were saved first. You can test this by creating a new column with a categorical variable `child`. `child` will take the value 1 in cases where age is <18, and a value of 0 in cases where age is >=18. 

To add this new variable you need to do two things (i) create a new column, and (ii) provide the values for each observation (i.e., row) based on the age of the passenger.

Adding a new column with Pandas in Python is easy and can be done via the following syntax:

```
your_data["new_var"] = 10
```

This code would create a new column in the `train` DataFrame titled `new_var` with `10` for each observation.

To set the values based on the age of the passenger, you make use of a boolean test inside the square bracket operator. With the `[]`-operator you create a subset of rows and assign a value to a certain variable of that subset of observations. For example,

```
train["new_var"][train["Survived"] == 1] = 0
```

would give a value of 0 to the variable `new_var` for the subset of passengers that survived the disaster.

*** =instructions

- Create a new column `Child` in the `train` data frame that takes the value `NaN`, if the passenger's age is `NaN`, `1` when the passenger is < 18 years and the value `0` when the passenger is >= 18 years. To create `NaN` use `float('NaN')`.
- Compare the normalized survival rates for those who are <18 and those who are older. Use code similar to what you had in the previous exercise.

*** =hint
Suppose you wanted to add a new column `clothes` to the `test` set and give all males the value `"pants"` and the others `"skirt"`:
```
test["clothes"] = "skirt"
test["clothes"][test["Sex"] == "male"] = "pants"
```


*** =pre_exercise_code

```{python}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sample_code

```{python}
# Create the column Child and indicate whether child or not a child. Print the new column.
train["Child"] = float('NaN')

# Normalized Survival Rates for under 18

# Normalized Survival Rates for over 18

```

*** =solution

```{python}
# Create the column Child, and indicate whether child or not a child. Print the new column.
train["Child"] = float('NaN')
train.Child[train.Age < 18] = 1
train.Child[train.Age >= 18] = 0
print(train.Child)

# Normalized Survival Rates for under 18
print(train.Survived[train.Child == 1].value_counts(normalize = True))

# Normalized Survival Rates for over 18
print(train.Survived[train.Child == 0].value_counts(normalize = True))

```

*** =sct
```{python}
msg = "Don't forget to set `normalize = True` when using `.value_counts()`."
test_function("print", 2,
              not_called_msg = msg,
              incorrect_msg = msg)

msg = "Compute the survival prportions for those OVER 18."
test_function("print", 3,
              not_called_msg = msg,
              incorrect_msg = msg)

success_msg("Well done! It looks like it makes sense to predict that all females will survive, and all men will die.")
```

--- type:NormalExercise lang:python xp:100 skills:2
## First Prediction

In one of the previous exercises you discovered that in your training set, females had over a 50% chance of surviving and males had less than a 50% chance of surviving. Hence, you could use this information for your first prediction: all females in the test set survive and all males in the test set die. 

You use your test set for validating your predictions. You might have seen that contrary to the training set, the test set has no `Survived` column. You add such a column using your predicted values. Next, when uploading your results, Kaggle will use this variable (= your predictions) to score your performance. 

*** =instructions
- Create a variable `test_one`, identical to dataset `test`
- Add an additional column, `Survived`, that you initialize to zero.
- Use vector subsetting like in the previous exercise to set the value of `Survived` to 1 for observations whose `Sex` equals `"female"`.
- Print the `Survived` column of predictions from the `test_one` dataset.

*** =hint
- To create a new variable, `y`, that is a copy of `x`, you can use `y = x`.
- To initialize a new column `a` in a dataframe `data` to zero, you can use `data['a'] = 0`.
- Have another look at the previous exercise if you're struggling with the third instruction.

*** =pre_exercise_code

```{python}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sample_code

```{python}
# Create a copy of test: test_one


# Initialize a Survived column to 0


# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
```

*** =solution

```{python}
# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female"
test_one["Survived"][test_one["Sex"] == "female"] = 1
print(test_one.Survived)
```

*** =sct

```{python}

test_function("print",
              not_called_msg = "Make sure to define the column `Survived` inside `test_one`",
              incorrect_msg = "Make sure you are assigning 1 to female and 0 to male passengers")

success_msg("Well done! If you want, you can already submit these first predictions to Kaggle [by uploading this csv file](http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/ch1_ex4_solution/my_solution.csv). In the next chapter, you will learn how to make more advanced predictions and create your own .csv file from Python.")
```

