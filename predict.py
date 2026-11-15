#!/bin/user/env python

import pickle
import numpy as np
from flask import Flask, request, jsonify


with open('models.bin', "rb") as f_in:
    (class_vectorizer, class_model, reg_vectorizer, reg_model) = pickle.load(f_in)


app = Flask('ping')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the startup data from the request as json
    startup = request.get_json()

    # Transform the data using DictVectorizer
    X_class = class_vectorizer.transform([startup])
    X_reg = reg_vectorizer.transform([startup])

    print("Class mapping:", class_model.classes_)

    # Predict the startup failure probablity using the model
    class_y_pred = class_model.predict_proba(X_class)[0, 1]

    # Decide whether the startup will fail or not
    failed = class_y_pred >= 0.5

    # Predict how much funding the startup could raise and apply exp and -1 to undo logarithm applied for regularization 
    reg_y_pred = np.expm1(reg_model.predict(X_reg)[0])
    

    if failed:
        decision = "Prediction: High risk of Poor Market Fit (Failed)"
    else:
        decision = "Prediction: Low risk of Poor Market Fit (Survived)"
    
    response = {
        "failure_probability": f'{round(class_y_pred * 100 , 3) }%',
        "failure": bool(failed),
        "predicted_funding_raised": f'${round(reg_y_pred, 3)}',
        "final_decision": decision 
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True)


