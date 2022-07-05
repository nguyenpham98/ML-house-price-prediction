Created a Machine Learning model to predict house price in California based on historical data using Random Forest Regression algorithm and Grid Search CV 
for optimization 

Tools: sklearn, pandas, numpy, matplotlib, pickle

Data source: https://www.kaggle.com/datasets/camnugent/california-housing-prices

My processing steps:

1/ Clean Data

There were missing values in total_bedrooms column that I filled in with SimpleImputer which is a class of sklearn that is used for filling in missing values. 
My strategy is calculating the mean values across columns and transform total_bedrooms column

2/ More features

There were already some useful features in the original dataset like median_house_value, median_income, total_rooms, etc... but I wanted to have some more extra features 
for better predictions. With total_rooms and households, I calculated rooms_per_house. I had the same process calculating population_per_house using population and 
households. As a result, the correlation between median_house_value does line up with rooms_per_house pretty well.

3/ Handling One Hot Encoding (OHE)

Since machine learning model really appreciate numerical values, I wanted to convert ocean_proximity of type string that describes how close a home is close to the ocean.
Luckily, pandas has a class get_dummies that basically separate those unique values and convert them into binary values (1 or 0)

4/ Split data for training

We have successfully cleaned our data. At this point, I used median_house_value as a good independent variable for prediction and split the datasets with train:test ratio
of 80:20. At first attempt to make prediction, Random Forest Regression got the following results:
mean: 50262.43034696649
std: 703.6384543708631

5/ Parameter tuning

I wanted to test if our model could use some optimization since we have too many parameters and features to truly know which one brings best prediction, so I wanted to
test with maybe 3,10,20,50 trees in our forest and 2,3,4,10 max features. Grid Search CV mostly figured out which combination is best through trial-and-error, at least 
to my understanding.

6/ Evaluation

I fed my model with the test dataset that got the following result: 
Mean Absolute Error: 32417.27453003876 
Mean Squared Error: 2415034265.049032 
Root Mean Squared Error: 49142.99812841126 
R2 score: 0.815703776061502


