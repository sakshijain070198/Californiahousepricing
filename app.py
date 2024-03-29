import pickle
from flask import Flask, request, jsonify,app, url_for, redirect, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

## Load the model
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    print(new_data)
    prediction=regmodel.predict(new_data)
    output=prediction[0]
    print(output)
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input= scalar.transform(np.array(data).reshape(1,-1))   
    print(final_input)
    prediction=regmodel.predict(final_input)
    output=prediction[0]
    return render_template('home.html',prediction_text='The Predicted house price is  is {}'.format(output))


if __name__=='__main__':
    app.run(debug=True)
