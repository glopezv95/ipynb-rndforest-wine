from auxiliary.functions import next, generate_header, dataset_import, df_export, pca_reduction, \
    variance_ratio_analysis, data_split, standard_scale, generate_model, predict

if __name__ == '__main__':
    
    generate_header()
    df, target, target_classes, df_describe, feature_names = dataset_import()
    
    print()
    df_export(df = df, name = 'features')
    df_export(df = target, name = 'target')
    df_export(df = df_describe, name = 'feature description')
    next()
    
    X_train, X_test, y_train, y_test = data_split(df = df, target = target)
    next()
    
    X_train, X_test, scaler = standard_scale(X_train = X_train, X_test = X_test)
    next()
    
    X_train, X_test, n_components, variance_ratio, pca = pca_reduction(X_train = X_train, X_test = X_test)
    next()
    
    variance_ratio_analysis(
        n_components = n_components,
        variance_ratio = variance_ratio,
        n_columns = len(df.columns)
        )
    
    model = generate_model(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test)
    
    next()
    
    while input('Predict values based on the model? (y/n) ').lower() == 'y':
        
        predict(
            model = model,
            key_features = feature_names,
            target_classes = target_classes,
            scaler = scaler,
            pca = pca)
        next()
        
    print()