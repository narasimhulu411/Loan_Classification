from flask import Flask,render_template,request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app=Flask(__name__)
ml_model=open('Random_Forest.pkl','rb')
Random_Forest_Model=joblib.load(ml_model)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method=='POST':
        try:
            principal_amt=int(request.form['Principal'])
            Terms=int(request.form['Terms'])
            past_due_days=int(request.form['past_due_days'])
            age=int(request.form['age'])
            education=int(request.form['education'])
            Gender=int(request.form['Gender'])
            pred_args=[principal_amt,Terms,past_due_days,age,education,Gender]
            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)

            predictions=Random_Forest_Model.predict(pred_args_arr)
        except ValueError:
            return "Please enter valid values"



        return render_template('predict.html',Predictions=predictions)

# We can write the code to get the values form html as below also
''' if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        to_predict_list_arr=np.array(to_predict_list)
        to_predict_list_arr=to_predict_list_arr.reshape(1,-1)
        predictions=KNN_Model.predict(to_predict_list_arr)'''

if __name__=="__main__":
    app.run(debug=True)
