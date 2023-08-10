import dill

from datetime import datetime
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier



def pipeline():
    import pandas as pd
    df = pd.read_csv('D:/dublin coding school lectures/python/final/classification/credit_customers.csv')

    X = df.drop('class', axis=1)
    y = df['class']

    numerical_features = make_column_selector(dtype_exclude=object)
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[('column_transformer', column_transformer)])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(max_iter=2000))
    ])

    pipe.fit(X, y)
    model_filename = (f'D:/dublin coding school lectures/python/final/classification/model/credit_customers_{datetime.now().strftime("%Y%m%d%H%M")}.pkl')

    with open(model_filename, 'wb') as file:
        dill.dump(pipe, file)


if __name__ == '__main__':
    pipeline()