import joblib
import pandas as pd

validation_model = joblib.load('../Models/model.pkl')
X_val = pd.read_csv('../Data/Splitted/X_val.csv')
y_val = pd.read_csv('../Data/Splitted/y_val.csv')


y_val_predict = validation_model.predict(X_val)
val_score =  (y_val_predict == y_val.values.ravel()).sum()/len(y_val)

print('Validation score is:')
print(val_score)
