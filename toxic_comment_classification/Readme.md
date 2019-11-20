ABEJA Platform sample code for text classification
===

## Description
### Train (Jupyter Notebook)
- use the [Wikipedia Talk Labels: Toxicity Dataset](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) in TSV format
- load the files in ABEJA Platform DataLake channel
- preprocess text data with CountVectorizer and TfidfTransformer
- train a Logistic Regression model and Gradient Boosting Decision Tree model in scikit-learn
### Predict
- predict if posted comments are toxic or non-toxic

## Requirements
- Python 3.6.x

## Docker
- abeja/all-cpu:19.04