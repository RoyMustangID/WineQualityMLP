import joblib

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from feature_config import used_feature

app = Flask(__name__)

## Model Loading
rfc_model = joblib.load('../Models/model.pkl')
used_column = used_feature

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input= pd.DataFrame([data], columns = used_column)
        
    #scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=rfc_model.predict(final_input)
    if output == 1:
        final_output = 'Good'
    else:
        final_output = "Bad"
    

    return render_template("home.html",prediction_text="The wine is {}".format(final_output))



if __name__=="__main__":
    app.run(debug=True)
   
     
