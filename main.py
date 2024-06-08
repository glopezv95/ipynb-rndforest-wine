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
    
    key_features, variance_ratio, pca_id = pca_reduction(df = df, feature_names = feature_names)
    next()
    
    variance_ratio_analysis(
        key_features = key_features,
        pca_id = pca_id,
        variance_ratio = variance_ratio,
        feature_names = feature_names)
    
    X_train, X_test, y_train, y_test = data_split(pca_id = pca_id, df = df, target = target)
    next()
    
    X_train_scaled, X_test_scaled = standard_scale(X_train = X_train, X_test = X_test)
    next()
    
    model = generate_model(
        X_train_scaled = X_train_scaled,
        X_test_scaled = X_test_scaled,
        y_train = y_train,
        y_test = y_test)
    
    next()
    
    while input('Predict values based on the model? (y/n) ').lower() == 'y':
        predict(model = model, key_features = key_features, target_classes = target_classes)
        next()
    print()