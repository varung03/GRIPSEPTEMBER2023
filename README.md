# GRIPSEPTEMBER2023
GRIPSEPTEMBER2023
THE SPARKS FOUNDATION - DATA DCIENCE & BUSINESS ANALYTICS INTERNSHIP

Author : Varun Gandhi

TASK I :- Prediction Using Supervised Learning (Level - Beginner) In this task , we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied . This is a simple linear regression task as it involves just two variables.

STEPS -
1) Importing the dataset
2) Visualizing the dataset
3) Data preparation
4) Training the algorithm
5) Visualizing the model
6) Making predictions
7) Evaluating the model
STEP 1 - Importing the dataset
# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
import seaborn as sns
# To ignore the warnings
import warnings as wg 
wg.filterwarnings("ignore")
​
# Reading data from remote Link
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
​
# now let's observe the dataset
df.head()
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
df.tail()
Hours	Scores
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
# To find the number of coliumns and rows
df.shape
(25, 2)
# To find more information about our dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 532.0 bytes
df.describe()
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000
# Now we will check if our dataset contains null or missing values
df.isnull().sum()
Hours     0
Scores    0
dtype: int64
As we see we do jot have null values in our dataset so we can now move on to our next step

STEP 2 - Visualizing the dataset

In this we will plot the data set to check wether we can opbserve any relation between the two variables or not

# Plotting the dataset
plt.rcParams["figure.figsize"]=[16,9]
df.plot(x='Hours',y='Scores', style='*', color='blue', markersize=10) 
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()

From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score. So, we can use the linear regression supervised learning model on it to predict further values

# we can also use .corr to determine the correlation between the variables
df.corr()
Hours	Scores
Hours	1.000000	0.976191
Scores	0.976191	1.000000
**STEP 3 - Data preparation**
​
In this step we will divide the data into "features"(inputs) and "labels"(outputs). After that we will split the whole dataset 
into 2 parts - testing and training data.
df.head()
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
# using iloc function we will divide the data
x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values
x
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])
y
array([[21],
       [47],
       [27],
       [75],
       [30],
       [20],
       [88],
       [60],
       [81],
       [25],
       [85],
       [62],
       [41],
       [42],
       [17],
       [95],
       [30],
       [24],
       [67],
       [69],
       [30],
       [54],
       [35],
       [76],
       [86]], dtype=int64)
# Splitting data into training and testig data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0) 
STEP - 4 Training the Algorithm
We have split our data into training and testing sets, and now is finally the time to train our algorithm.

from sklearn.linear_model import LinearRegression
​
model = LinearRegression() 
model.fit(X_train, y_train)

LinearRegression
LinearRegression()
STEP 5 - Visualizing The Model
Now that we have trained our algorithm, it's time to make some predictions.

line = model.coef_*x + model.intercept_
​
# Plotting for the training data
plt.rcParams["figure.figsize"] = [16,9] 
plt.scatter(X_train, y_train, color='red')
plt.plot(x, line, color = 'green');
plt.xlabel('Hours Studied') 
plt.ylabel('Percentage Score')
plt.grid()
plt.show()

**STEP 6 - Making predictions**
​
Now that we have trained our algorithm, it is time to make some predictions.
print(X_test) # Testing data - In Hours
y_pred = model.predict(X_test) # Predicting the scores
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
# Compairing Actual vs Predicted
y_test
array([[20],
       [27],
       [69],
       [30],
       [62]], dtype=int64)
y_pred
array([[16.88414476],
       [33.73226078],
       [75.357018  ],
       [26.79480124],
       [60.49103328]])
# Compairing Actual vs Predicted
comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred]})
comp
Actual	Predicted
0	[[20], [27], [69], [30], [62]]	[[16.884144762398037], [33.73226077948984], [7...
# Testing with your own data
​
hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours s",own_pred[0])
The predicted score if a person studies for 9.25 hours s [93.69173249]
STEP - 7 Evaluating the model
The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
Mean Absolute Error: 4.183859899002975
