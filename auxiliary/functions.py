import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly import express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from .const import TITLE, SUBTITLE, RANDOM_STATE

def next():
    print('--------------------------------------------------------------------------------')

def generate_header():
    
    print()
    print(TITLE)
    next()
    print(SUBTITLE)
    next()
    print()

def feature_description(df: pd.DataFrame) -> pd.DataFrame:    
    """
    Describe the features of the dataset
    """    
    df_out = df.describe()
    
    print('Feature analysis:')
    print()
    print(df_out)
    
    return df_out

def dataset_import():
    
    bool = input('Import dataset? (y/n) ')
    if bool.lower() == 'y':
        
        data = load_wine(as_frame = True)

        df = pd.DataFrame(data['data'])
        target = data['target']
        feature_names = data['feature_names']
        target_classes = data['target_names']
        num_observations = len(df)
        df_info = df.info()

        print()
        print('Dataset loaded succesfully')
        print()
        print(f'DataFrame info ({num_observations} observations)')
        print()
        print(df_info)
        print()
        print('Null feature values:', df.isna().any().any())
        print('Possible target values:', target_classes)
        next()
        
        df_describe = feature_description(df = df)
        print(df_describe)
        
        return [df, target, target_classes, df_describe, feature_names]
    
    else:
        print()
        exit()
        
def df_export(df: pd.DataFrame, name:str) -> None:
    
    bool = input(f'Export dataset {name}? (y/n) ')
    
    if bool.lower() == 'y':
        df.to_csv(f'{name}.csv')
        
def variance_ratio_analysis(
    key_features: list, pca_id:int, variance_ratio:np.ndarray, feature_names: list):
    
    fig = px.bar(x = key_features, y = variance_ratio[:pca_id])
    fig.add_bar(x = feature_names[pca_id:], y = variance_ratio[pca_id:], name ='excluded')

    fig.update_layout(
        title = 'Variance ratio per feature (scaled)',
        plot_bgcolor = 'rgba(0,0,0,0)',
        font = dict(family = 'sans-serif'))

    fig.update_xaxes(title = '')
    fig.update_yaxes(title = '')
    fig.update_traces(hovertemplate = '<b>Feature:</b> %{x}<br><b>Variance ratio:</b> %{y:.3f}')
    fig.show()
        
def pca_reduction(df: pd.DataFrame, feature_names: list):
    
    bool = input('Next step involves applying PCA dimension reduction. Proceed? (y/n) ')
    if bool.lower() == 'y':
        
        pca_scaler = StandardScaler()
        X_scaled = pca_scaler.fit_transform(df.values)

        pca = PCA()
        pca.fit(X_scaled)
        variance_ratio = pca.explained_variance_ratio_

        pca_features_cum_dict = {}

        for i in range(len(feature_names)):
            pca_features_cum_dict[i] = sum(variance_ratio[:i +1])

        threshold = 0.9
        var_cum_sum = 0
        pca_id = 0

        for i in range(len(pca_features_cum_dict)):
            if var_cum_sum <= threshold:
                var_cum_sum = list(pca_features_cum_dict.values())[i]
                pca_id = i -1

        key_features = feature_names[:pca_id]
        print(f'PCA dimension reduction applied. {pca_id} features kept.')
    
        return [key_features, variance_ratio, pca_id]
    
    else:
        print()
        exit()

def data_split(pca_id:int, df:pd.DataFrame, target: pd.Series):
    
    bool = input('Next step involves applying a train/test split of the data. Proceed? (y/n) ')
    if bool.lower() == 'y':
        
        X = df.iloc[:, :pca_id]
        test_size = 0.3

        X_train, X_test, y_train, y_test = train_test_split(
            X, target,
            test_size = test_size,
            random_state = RANDOM_STATE,
            stratify = target)

        print('Train/test split applied.')
        print()
        print(f'Array lengths (test size: {test_size}, random state: {RANDOM_STATE}):')
        next()
        print('Training features:\t', len(X_train))
        print('Training target:\t', len(y_train))
        print('Testing features:\t', len(X_test))
        print('Testing target:\t\t', len(y_test))
        
        return [X_train, X_test, y_train, y_test]
    
    else:
        print()
        exit()
    
def standard_scale(X_train:np.ndarray, X_test: np.ndarray):
    
    bool = input('Next step involves applying a Standard Scaler to normalise the features. Proceed? (y/n) ')
    if bool.lower() == 'y':
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print('StandardScaler applied.')
        
        return [X_train_scaled, X_test_scaled]
    
    else:
        print()
        exit()

def generate_model(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    
    bool = input('Next step involves generating the optimised model (Random Forest Regressor). Proceed? (y/n) ')
    if bool.lower() == 'y':
        
        n_estimators = 500
        model = RandomForestClassifier(n_estimators = n_estimators)
        model.fit(X_train_scaled, y_train)
        score = round(model.score(X_test_scaled, y_test), 3)

        print()
        print('Chosen model: Random Forest Classifier')
        print(f'Key hyperparameter: n_estimators = {n_estimators}')
        print('Model mean accuracy:', score)
        
        return model
    
    else:
        print()
        exit()
        
def predict(model: RandomForestClassifier, key_features: list, target_classes:list):
        
        X = []
        
        for feature in key_features:
            val = float(input(f'Value for feature {feature}: '))
            X.append(val)
            
        prediction = model.predict(X = np.array([X]))
        print('Predicted value:', target_classes[prediction[0]])