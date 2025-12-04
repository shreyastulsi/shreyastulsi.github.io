<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Project Overview

In this project, I applied various ML techniques to demonstrate the Atmospheric and Oceanic Studies C111 course at UCLA, to study the impacts of both demographic factors and lifestyle choices on an individual's lifetime medical charges.


***

## Introduction 

Health care costs have risen dramatically over the past few decades, making it more important than ever for both policy issuers and insurance providers to understand the key contributors, as well as overall emerging trends. Specifically, the goal of this project is to provide a predictive framework that, when given demographic and lifestyle characteristics about an individual, can give a reasonable estimate of their lifetime insurance cost. In this project, I used a dataset publicly available from Kaggle, which considered the following attributes: age, sex, bmi, #of children, smoker, region when predicting the target variable charges( individual medical insurance costs in USD). This setup naturally caused me to pursue a supervised ml regression problem, where the target became the continuous variable charges. I sampled several regression approaches: Linear regression, Random Forest, Gradient Boosting, and even a Neural Network(MLP). These approaches were then evaluated on common ML metrics such as (R^2), mean absolute error(MAE), root mean squared error(RSME), and mean absolute percentage error(MAPE).

Through these experiments, my final conclusion was that the Random Forest approach performed the best, and offered the best predictive abilities for the problem statement out of any of the approaches. Furthermore, the key contributors were concluded as smoker status, BMI, and age, with smoking being the clear leading cause. 

## Data Specifications

Title: “Medical Cost Personal Datasets” insurance dataset on Kaggle

Columns
	age - age of the insurance beneficiary(years)
	Sex - male or female
	cmi- body mass index
	children- number of dependants that are covered by insurance
	smoker- yes/no indication for smoking status
	region: residential region, northeast, southeast, southwest, northwest
	charges: individual medical costs billed by health insurance(USD)

When performing an initial analysis of the data, I computed the following statistics:


Number of samples: 1,338
Features: age, sex, bmi, children, smoker, region
Target: charges
Average charges: $13,270.42
Standard deviation of charges: $12,110.01
Minimum charges: $1,121.87
Maximum charges: $63,770.43

Before modeling, I performed the following preprocessing scripts:

```python
def preprocess_data(df):
        df_processed = df.copy()
    categorical_cols = ["sex", "smoker", "region"]
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    X = df_processed.drop("charges", axis=1).values
    y = df_processed["charges"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


```

Printed the datatypes, missing values counts to ensure a consistent schema(no errant values)
Categorical encoding: for the non numerical datatypes(specifically sex, smoker and region, these were encoded using the Label Encoder)
Train-test split: Split dataset into train and test datasets with 80/20 ratio, using train_test_split and a fixed random_state = 42
Used StandardScaler for input features, so Linear Regression and ML would work better


I then also performed further exploratory data analysis to visualize the relationships within my dataset in a clearer fashion.
<figure>
  <img src="assets/IMG/correlation_heatmap.png" width="500" alt="Correlation heatmap">
  <figcaption>Figure 1: Correlation heatmap of numeric features.</figcaption>
</figure>


![](assets/IMG/eda_visualizations.png)



## Modeling Specifications


Supervised regression task
Given feature vector(x) describing person, predict continuous outcome(y)

Models Used
Linear Regression
Random Forest 
Gradient Boosting
Neural Networks(MLP)


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42, max_depth=5
    ),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)

```
Evaluation Metrics
Coefficient of determination(R^2)
Mean Absolute Error(MAE)
Root Mean Squared Error(RMSE)
Mean Absolute Percentage Error(MAPE)





Data Results and Analysis :

Random Forest
(R^2 = 0.8645)
MAE = $2,517.11
RMSE = $4,586.63
MAPE = 30.11%
Gradient Boosting
(R^2 = 0.8612)
MAE = $2,485.65
RMSE = $4,642.81
MAPE = 29.50%
Neural Network (MLP)
(R^2 = 0.8040)
MAE = $3,898.93
RMSE = $5,515.77
MAPE = 46.38%
Linear Regression
(R^2 = 0.7833)
MAE = $4,186.51
RMSE = $5,799.59
MAPE = 47.09%







From these results, we can see the non-linear models are all outperforming the linear regression, across all metrics that we are tracking. The most promising results are from the random forest approach, giving us a strong R^2 metric of 0.8645, as well as the lowest RMSE. It is important to note however, that Gradient Boosting did have a lower MAPE and a lower MAE, thus making gradient boosting and random forest the two winners. 

Exploring the random forest further however I was curious to better understand its rationale, so I examined its ranked feature importances.

smoker: 0.6168
bmi: 0.2124
age: 0.1334
children: 0.0187
region: 0.0127
sex: 0.0059

Furthermore here is also the plot that demonstrates how the predicted vs.actual charges are scattered for the random forest approach. From visual inspection it becomes clear that the model is indeed tracking actual values at a reasonably high level. Furthermore, we can see that residuals are centered around zero, and the major errors occur when the cost is being predicted is for abnormally high-cost individuals.

![](assets/IMG/model_comparison_metrics.png)
![](assets/IMG/best_model_analysis_Random_Forest.png)
![](assets/IMG/feature_importance.png)




Takeaways

Tree-based ensemble methods outperform the linear regression and MLP configurations. Best model achieves as R^2 of 0.8645, meaning it can capture meaningful signal at a fairly high rate, but a MAPE of 30% demonstrates clear prediction possibilities, especially for individuals with very high charges
Smoking across the board is the most important feature( .62 importance in random forest approach) in the ML analysis. This is also demonstrated when isolating for smokers versus no smokers, an average of $32k to $8.4k respectively.
BMI,  age come after smoking in terms of factor importance, .21 and .13 respectively. High BMI old age can clearly be argued to contribute to higher charges within insurance.
Children, region, sex are contributing relatively little, at least in the limited analysis we have conducted in this project.

Limitations

Size of this dataset is somewhat limited(only 1338 samples) thus may not generalize well to other populations
The evaluation is being done using a simple train-test split system. Performing cross-validation, as well as more advanced techniques such as hyperparameter tuning, could boost the performance of the ML approaches we are employing
The preprocessing being doen is quite straightforward, the results would probably improve given more advanced feature engineering(this includes things like non-linear transformations)




