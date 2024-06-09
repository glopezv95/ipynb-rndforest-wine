import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly import express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
    n_components: int,
    variance_ratio: np.ndarray,
    n_columns: int):
    
    fig = px.bar(x = [f'PC{n}' for n in range(n_components)], y = variance_ratio[:n_components])
    fig.add_bar(x = [f'PC{n}' for n in range(n_components, n_columns)], y = variance_ratio[n_components:], name ='excluded')

    fig.update_layout(
        title = 'Variance ratio per Principal Component (scaled)',
        plot_bgcolor = 'rgba(0,0,0,0)',
        font = dict(family = 'sans-serif'))

    fig.update_xaxes(title = '')
    fig.update_yaxes(title = '')
    fig.update_traces(hovertemplate = '<b>Principal Component:</b> %{x}<br><b>Variance ratio:</b> %{y:.3f}')
    fig.show()
        
def pca_reduction(X_train:np.ndarray, X_test: np.ndarray) -> None:
    
    bool = input('Next step involves applying PCA dimension reduction. Proceed? (y/n) ')
    if bool.lower() == 'y':

        pca = PCA()
        pca.fit_transform(X_train)
        pca.transform(X_test)

        variance_ratio = pca.explained_variance_ratio_

        pca_features_cum_dict = {}

        for i in range(len(pca.components_)):
            pca_features_cum_dict[i] = sum(variance_ratio[:i +1])

        threshold = 0.9
        var_cum_sum = 0
        n_components = 0

        for i in range(len(pca_features_cum_dict)):
            if var_cum_sum <= threshold:
                var_cum_sum = list(pca_features_cum_dict.values())[i]
                n_components = i -1

        pca = PCA(n_components = n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.fit_transform(X_test)

        print(f'PCA dimension reduction applied. {n_components} components kept')
        
        return [X_train, X_test, n_components, variance_ratio, pca]
    
    else:
        print()
        exit()

def data_split(df:pd.DataFrame, target: pd.Series):
    
    bool = input('Next step involves applying a train/test split of the data. Proceed? (y/n) ')
    if bool.lower() == 'y':
        
        X = df.values
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
        
        return [X_train_scaled, X_test_scaled, scaler]
    
    else:
        print()
        exit()

def generate_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    
    bool = input('Next step involves generating the optimised model (K Nearest Neighbors). Proceed? (y/n) ')
    if bool.lower() == 'y':
        
        n_neighbors = 7
        model = KNeighborsClassifier(n_neighbors = n_neighbors)
        model.fit(X_train, y_train)
        score = round(model.score(X_test, y_test), 3)

        print()
        print('Chosen model: K Nearest Neighbors Classifier')
        print(f'Key hyperparameter: n_neighbors = {n_neighbors}')
        print('Model mean accuracy:', score)
        
        return model
    
    else:
        print()
        exit()
        
def predict(
    model: KNeighborsClassifier,
    key_features: list,
    target_classes:list,
    scaler: StandardScaler,
    pca: PCA):
        
        X = []
        
        for feature in key_features:
            val = float(input(f'Value for feature {feature}: '))
            X.append(val)
        
        X = scaler.transform(X = np.array([X]))
        X = pca.transform(X = X)
        prediction = model.predict(X = X)
        
        print()
        print('Predicted target value:', target_classes[prediction[0]])