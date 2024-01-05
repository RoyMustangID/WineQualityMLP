import pandas as pd
import joblib
from imblearn.under_sampling import TomekLinks
from check_iqr import check_iqr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# Import data
X_train = pd.read_csv('../Data/Splitted/X_train.csv')
y_train = pd.read_csv('../Data/Splitted/y_train.csv')


# Set outlier limit
X_column_name = X_train.columns.values.tolist()
upper_limit = []
lower_limit = []
upper_outlier = []
lower_outlier =[]
for cols in X_column_name:
    upper_limit, lower_limit = check_iqr(X_train, cols,3 )
    upper_outlier.append(upper_limit)
    lower_outlier.append(lower_limit)

upper_outlier[X_column_name.index('chlorides')] = 0.605
upper_outlier[X_column_name.index('volatile_acidity')] = 1
upper_outlier[X_column_name.index('fixed_acidity')] = 15


# Drop outlier
for columns in X_column_name:
    y_train.drop(X_train[X_train[columns] > upper_outlier[X_column_name.index(columns)]].index, inplace = True)
    X_train.drop(X_train[X_train[columns] > upper_outlier[X_column_name.index(columns)]].index, inplace = True)
    y_train.drop(X_train[X_train[columns] < lower_outlier[X_column_name.index(columns)]].index, inplace = True)
    X_train.drop(X_train[X_train[columns] < lower_outlier[X_column_name.index(columns)]].index, inplace = True)


# Missing Value Handling
for column in X_column_name:
    X_train[column] = X_train[column].fillna(X_train[column].median())



#Feature Selection

used_column = ['fixed_acidity',
 'volatile_acidity',
 'citric_acid',
 'residual_sugar',
 'chlorides',
 'free_sulfur_dioxide',
 'pH',
 'sulphates',
 'alcohol']

X_train = X_train[X_train.columns.intersection(used_column)]

# Undersampling the majority category in target
TL = TomekLinks(sampling_strategy = 'majority', n_jobs = -1)
X_balance, y_balance = TL.fit_resample(X_train,y_train)


# Pipeline

# Used Model
best_model = RandomForestClassifier(criterion = 'gini',
                                    max_depth = 25,
                                    min_samples_leaf = 1,
                                    min_samples_split = 5,
                                    n_estimators = 90)



# Feature Pipeline
feature_pipeline = Pipeline(
    steps=[
        ("computer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler())
    ]
)

# Transformer
feature_preprocessor = ColumnTransformer(
    transformers=[("num", feature_pipeline, used_column)],
    remainder='drop')

# Pipeline Model
pipe = Pipeline(
    steps=[("preprocessor", feature_preprocessor), ("classifier", best_model)]
)

pipe.fit(X_balance, y_balance.values.ravel())



# Export model
joblib.dump(pipe, "model.pkl")

y_predict = pipe.predict(X_balance)
print('This model accuracy towards itself is:')

print((y_predict == y_balance.values.ravel()).sum()/len(y_balance))