## Adult Census Income Prediction
<b>
This study explores the application of machine learning and data mining techniques as a means to address the challenge of income inequality. This project was undertaken as part of an internship with "iNeuron Intelligence Pvt Ltd."
</b><br> <br>

## Objectives
The Goal is to predict whether a person has an income of more than 50K a year or not.

This is basically a binary classification problem where a person is classified into the 

>50K group or <=50K group.<br><br>

## Life Cycle of Machine Learning Project
Life Cycle of implementing machine learning application.
- Gathering the Data
- Data Preparation
- Data Preprocessing
- Create Model
- Evaluate Model
- Deploy the model
<br>

## Dataset
The UCI Adult Dataset has been used for this purpose, taken from the Kaggle. link is below.

- [datset](https://www.kaggle.com/datasets/overload10/adult-census-dataset?resource=download)

## Requirements
* Python (Programming Language version 3.7+)
* Flask (Python Backend Framework)
* sklearn (Machine Learning Library)
* pandas (Python Library for Data operations)
* NumPy (Python Library for Numerical operations)
* imblearn (sampling Library)
* VS code (IDE)
<br><br>

#### How to run this code...
- Create virtual environment
```bash
conda create -n myenv python=3.8
```
- Activate the environment
```bash
conda activate myenv
```
- Install the packages
```bash
pip install -r requirements.txt
```
- Run the app
```bash
python app.py
```
- Navigate to URL http://127.0.0.1:5000/
<br>

- Enter valid values in all input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary Class on the HTML page!
