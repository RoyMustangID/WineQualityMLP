import joblib

from flask import Flask,request,app,render_template
import numpy as np
import pandas as pd
from feature_config import used_feature
from feature_config import min_limit
from feature_config import max_limit

app = Flask(__name__)

## Model Loading
rfc_model = joblib.load('Models/model.pkl')
used_column = used_feature
lower_limit = min_limit
upper_limit = max_limit

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    #transform the data
    data=[float(x) for x in request.form.values()]
    final_input= pd.DataFrame([data], columns = used_column)
        
    #check whether the data are in range
    input_data = "OK"
    for column in used_column:
        if final_input[column][0] > upper_limit[column]:
            input_data = "OVER"
        elif final_input[column][0] < lower_limit[column]:
            input_data = "UNDER"
    

    if input_data == "OVER":
        prediction = "Some values are over regulation threshold. Prediction cannot be done"
    elif input_data == "UNDER":
        prediction = "Some values are below regulation threshold. Prediction cannot be done"
    else:
        # If the imputed data are ok, prediction can be done
        output=rfc_model.predict(final_input)
        if output == 1:
             final_output = 'good'
        else:
             final_output = "bad"
        prediction="The wine is {}".format(final_output)

    return render_template("home.html",prediction_text = prediction)



if __name__=="__main__":
    app.run(debug=True)
   
     
