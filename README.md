# SC1015 FCMB Team 3 Flight Delay Prediction
### Motivation
For most people, buying airline flights is a lengthy process that requires much commitment. It seems no matter how well one plans, there are always seemingly random delays that can interfere with long awaited plans. We have set out to see if there is some way of accurately predicting a flight delay.

### Project Goal
   
This project aims to accurately predict the probability of whether a flight will be delayed or not (classification), based on data that would be available during the time of scheduled flight.

# Project Setup

Please view the notebook in the following order:

### [1. Data Collection and Preparation](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/1_Data_Preparation.ipynb)

### [2. Exploratory Data Analysis (EDA) and Visualization](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/2_Exploratory_Data_Analysis.ipynb)

### [3. Data Processing](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/3_Data_Processing.ipynb) 

### [4. Training Models](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/4_Training_Models.ipynb) 

# Dataset Used

**Note that only the "Flight Status Prediction" was used in the final training of models.**
 The "Weather Events" and "IATA Codes Dataset" were only used for analysis purposes and were eventually dropped.
 As such, only `dataset.csv` is required for Notebooks 3 & 4 and included in the repository. Please download the files `WeatherEvents_Jan2016-2022.csv` and `iata-icao.csv` separately from the sources below.

- **"Flight Status Prediction"** by *Rob Mulla (Primary)*  
  - https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022/data
- **"US Weather Events (2016 - 2022)"** by *Sobhan Moosavi*  
  - https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events
- **"IATA Codes Dataset"** 
  - https://github.com/ip2location/ip2location-iata-icao


# Learning Points

### [1. Data Collection and Preparation](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/1_Data_Preparation.ipynb)

- Cleaned raw data obtained from primary sources and reduced dataframe sizes.
- Added additional from derived data, and pruned unecessary columns where necessary.
- Feature engineered "SevereWeatherEvents" column to draw correlations with weather events and flight predictions

### [2. Exploratory Data Analysis (EDA) and Visualization](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/2_Exploratory_Data_Analysis.ipynb)

Visualized flight distribution to determine the relation between variables.
- **Departure Delay Analysis**
  - Univariate Analysis
    - Boxplot, Histogram, Violinplot
    - Barplot
  - Analyzed central tendency, spread, and skewness
    - Identified and removed outliers


- **Numerical Variables Analysis**
  - Univariate Analysis
    - Boxplot, Histogram, Violinplot
  - Bivariate Analysis
    - Pairplot, Correlation Matrix Heatmap
  - Reinterpretation of Numerical Variables to Categorical
  

- **Categorical Variables Analysis**
  - Univariate Analysis
    - Line plot, Bar plot
  - Bivariate Analysis
    - Chi-square Test
  


### [3. Data Processing](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/3_Data_Processing.ipynb)

- **Resampling Data**
  - Reduced the size of the dataset by random downsampling to handle large volumes of data.
  - Maintained data integrity while reducing computational load for modeling.
  
  
- **Encoding Categorical Variables**
  - Transformed categorical variables into numerical representations for machine learning algorithms.
  - Implemented one-hot encoding to represent categorical variables as binary vectors.
  - Expanded categorical features into multiple binary columns to capture all levels of categories.

### [4. Training Models](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/4_Training_Models.ipynb) 

- **Models Used** <br><br>
Given the myrad of machine learning models, we chose the most popular and suitable model for performing binary classification.

  - Decision Tree Classifier
    - Uses a single decision tree, simple to visualise decision boundaries
  - Random Forest Classifier
    - Ensemble learning by combining multiple decision trees, more robust to overfitting
  - Logistic Regression
    - Linear model used for binary classification


- **Evaluation Metrics** <br><br>
Evaluation of models can be misleading if a wrong metric was used. This was the case when we achieved high accuracy at some point although it was due to overfitting. 
In this problem, we prioritise **recall** to minimize cases of false negatives where by overpreparing passengers for a delay.

    - Recall (**Most Important**)
    - Accuracy
    - Confusion Matrix
    - Precision
    - F1-Score
    - AUC-ROC


- Removing Irrevelant Features with `feature_importance_score`
- Hyperparameter Tuning with `GridSearch` and `k-Fold Cross Validation`

# Conclusion

![Results](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/images/results.png?raw=true)

![Features](https://github.com/ElsonNg/SC1015_Data_Science/blob/main/images/features.png?raw=true)

Our best model trained on a `DecisionTreeClassifier` achieved a best recall of `84.15%` after hyperparameter tuning. While this is far from ideal to be used in reality, due to the nature of external events affecting flight delays, augmenting the dataset with real-time forecasts like weather forecast and service disruption information will likely improve the model's performance.

# Contributions

Ng Yuan Da Elson (@ElsonNg)  
 - Data Processing
 - Training Models
 
Wu Yiqing (@ishowscript)    
 - Exploratory Data Analysis  
 - Data Preparation  
 
Eshkenazi Jacques (@JackEshkenazi)  
 - Exploratory Data Analysis  
 - Data Preparation  

# References

1. https://www.geeksforgeeks.org/chi-square-test-for-feature-selection-mathematical-explanation/
2. https://towardsdatascience.com/statistical-learning-ii-data-sampling-resampling-93a0208d6bb8/
3. https://medium.com/@denizgunay/random-forest-af5bde5d7e1e/
4. https://www.natasshaselvaraj.com/logistic-regression-explained-in-7-minutes/
5. https://www.scaler.com/topics/machine-learning/grid-search-in-machine-learning/
6. https://www.analyticsvidhya.com/blog/2022/02/a-comprehensive-guide-on-hyperparameter-tuning-and-its-techniques/
7. https://mljar.com/blog/feature-importance-in-random-forest/
