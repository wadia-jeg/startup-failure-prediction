# Startup Failure & Funding Prediction Service

## 1. Problem Description
Startups operate in high-risk environments where early strategic missteps—such as poor market fit, heavy competition, or monetization failure—frequently lead to shutdowns despite substantial capital injections. 

The goal of this project is to build and deploy an end-to-end Machine Learning system that solves two critical questions for founders, angel investors, and venture analysts evaluating early-stage companies:
1. **Risk Classification:** Predict the probability that a startup will fail using the *Poor Market Fit* as target variable.
2. **Funding Estimation (Regression):** Estimate how much total capital (in USD) a startup profile is likely to raise using *How Much Raised* feature.

**How the Solution is Used:**
The trained models are packaged into a Flask web service and REST API. Investors or founders can submit a startup's operational profile and risk indicators via an HTTP `POST` request to receive an immediate risk assessment, failure probability percentage, and expected funding valuation.

---

## 2. Dataset & Reproducibility
The project uses the **`Startup Failures.csv`** dataset, which contains historical records of failed startups across multiple sectors, detailing their operational duration, funding raised, and specific failure drivers.

### Reproducibility Setup
* The dataset (`Startup Failures.csv`) is committed directly inside the repository under the project directory so all notebooks and scripts run out-of-the-box.
* All random splits, cross-validation folds, and model initializations use fixed `random_state` seeds to ensure 100% deterministic results.

---

## 3. Exploratory Data Analysis (EDA)
Comprehensive EDA was performed in `notebook.ipynb` prior to model selection:
* **Data Cleaning & Regex Parsing:** Cleaned raw currency strings in funding columns and extracted operational lifespans from `Years of Operation`.
* **Missing Values & Ranges:** Inspected min/max ranges, distributions, and handled null values across categorical and binary indicator columns.
* **Target Variable Analysis:** Analyzed class balance for the binary failure target and applied a logarithmic transformation (`np.log1p`) to normalize the right-skewed funding target variable.
* **Feature Importance & Correlation:** Evaluated mutual information scores and correlation matrices to identify the strongest predictors of failure risk and capital raised.

---

## 4. Model Training & Hyperparameter Tuning
Both **linear** and **tree-based** algorithms were trained, evaluated, and tuned in `notebook.ipynb`:
* **Classification Models (Failure Risk):** Logistic Regression, Decision Tree, Random Forest (tuned via ROC-AUC).
* **Regression Models (Funding Raised):** Linear Regression, Decision Tree, Random Forest, (tuned via RMSE on log-transformed values).

---

## 5. Exporting Notebook to Training Script
All data preprocessing, feature vectorization (`DictVectorizer`), model training, and serialization logic from the notebook has been exported into a standalone script: **`train.py`**.

To re-train both models from scratch and generate the serialized artifact (`models.bin`), run:
```bash
python train.py

```
---

## 6. Dependency & Environment Management

This project uses **Pipenv** to isolate dependencies and guarantee exact package versions via `Pipfile` and `Pipfile.lock`.

### Installation & Activating the Virtual Environment

1. **Install Pipenv**:
```bash
pip install pipenv

```


2. **Install exact dependencies**:
```bash
pipenv install --deploy

```


3. **Activate the virtual environment**:
```bash
pipenv shell

```

---

## 7. Containerization (Docker)

The Flask prediction service is fully containerized using the included **`Dockerfile`**.

### Build the Docker Image

```bash
docker build -t startup-predictor .

```

### Run the Docker Container

```bash
docker run -it -p 8088:8088 startup-predictor

```
### Send a Request 

```bash

curl -X POST "http://127.0.0.1:8088/predict" \
     -H "Content-Type: application/json" \
     -d '{
        "sector": "Information",
        "years_of_operation": 3,
        "giants": 0,
        "no_budget": 1,
        "competition": 1,
        "acquisition_stagnation": 0,
        "platform_dependency": 0,
        "monetization_failure": 0,
        "niche_limits": 0,
        "execution_flaws": 0,
        "trend_shifts": 0,
        "toxicity_trust_issues": 0,
        "regulatory_pressure": 0,
        "overhype": 0
        }

```

---

## 8. Model Deployment & Usage

### Option A: Deploying on PythonAnywhere 

Because PythonAnywhere does not support Docker and runs from a system root directory (`/var/www/`), absolute path resolution is implemented in `predict.py` to seamlessly load `models.bin`.

1. **Sign up / Log in** to [PythonAnywhere](https://www.pythonanywhere.com/).
2. **Create a Flask Web App** via the **Web** tab (choose Python 3.x and Flask).
3. **Upload Files:** Upload `predict.py` and `models.bin` into your project directory (e.g., `/home/yourusername/mysite/`).
4. **Configure Absolute Paths in `predict.py`:** Ensure your model loader uses absolute paths:
    ```python
      import os

      current_dir = os.path.dirname(os.path.abspath(__file__))
      model_path = os.path.join(current_dir, 'models.bin')

      with open(model_path, 'rb') as f_in:
        class_vectorizer, class_model, reg_vectorizer, reg_model = pickle.load(
            f_in
        )

      ```

5. **Configure WSGI File:** In the PythonAnywhere **Web** tab, update your WSGI configuration file:
   ```python
   import sys

   path = '/home/yourusername/mysite'
   if path not in sys.path:
     sys.path.append(path)

   from predict import app as application



6. **Install Packages & Reload:** Open a Bash console and install dependencies if needed (`pip install --user scikit-learn pandas numpy flask`), then click the green **Reload** button on your PythonAnywhere Web dashboard.

---

### Making Predictions via `curl`

Send a `POST` request to your live endpoint:

```bash
curl -X POST "https://wadiej.pythonanywhere.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
        "sector": "Information",
        "years_of_operation": 3,
        "giants": 0,
        "no_budget": 1,
        "competition": 1,
        "acquisition_stagnation": 0,
        "platform_dependency": 0,
        "monetization_failure": 0,
        "niche_limits": 0,
        "execution_flaws": 0,
        "trend_shifts": 0,
        "toxicity_trust_issues": 0,
        "regulatory_pressure": 0,
        "overhype": 0
        }

```

### Example JSON Response

```json
{   
    "failure":false,
    "failure_probability":"41.153%",
    "final_decision":"Prediction: Low risk of Poor Market Fit (Survived)",
    "predicted_funding_raised":"$5723312.117"}

```

