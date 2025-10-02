# project-2 (draft)

## Introduction

Many companies face the challenge of customers leaving their product for a competitor. For example, Adobe has lost customers to Figma. This situation can put businesses at risk and impact their growth. To address this, I wanted to explore whether it is possible to predict customer churn based on various factors collected by a company about its customers.

## Data Selection

For this project, I am using the Iranian Churn Dataset from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset). The dataset contains 3,150 customer records with 13 features, including call failures, complaints, subscription length, charge amount, seconds of use, frequency of use, frequency of SMS, distinct called numbers, age group, service type, and customer value. All attributes are aggregated over the first 9 months, while the churn label reflects the customer’s status at the end of 12 months, leaving a 3-month planning gap.

## Data Preprocessing

### Handling Categorical Values

Many machine learning algorithms, including Random Forests, work only with numbers, not text. So categorical columns (like "gender": ["Male", "Female"]) need to be converted to numeric form. So, pd.get_dummies converts categorical variables into dummy/indicator variables (One-Hot Encoding)

### Separate Churn

In supervised learning, we need to separate Independent variables that describe the customer (age, usage, complaints, etc) from what we want to predict. Splits the dataset into X where all columns except "churn" and y with "churn" column.

### Train-Test Split

We split into training (80%) and testing (20%) sets so we can evaluate how well the model generalizes to unseen data. we need to ensure that the same churn/non-churn ratio in both train and test sets (important for imbalanced datasets).

## Data visualization

This is the data corelation with data among the table attributes.

<img src = "./images/heatmap.png" width=300 height=300 style="text-align:center;">

Shows relationships between dataset features and churn.

## Modeling