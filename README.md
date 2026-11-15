# Startup Failure Prediction API

## Description of the Problem
The goal of this project is to building and deploying 2 machine learning models one for `classification` capable of predicting whether a startup is likely to succeed or fail based on the `poor_market_fit` feature, and for `regression` capale of predciting the toral of funding a startup might get based on `how_much_fund_raised` feature. Providing early risk assessments helps investors and founders identify critical risk factors and make data-driven decisions.

## Instructions on how to run the project

### 1. Prerequisites
Ensure you have Python and Pipenv installed on your local machine:
* Python 3.13+
* Pipenv

### 2. Local Installation & Execution
Clone the repository and install the required dependencies using Pipenv:
```bash
pipenv install 

pipenv shell

python predict.py