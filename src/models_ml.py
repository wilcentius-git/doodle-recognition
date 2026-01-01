from sklearn.ensemble import RandomForestClassifier

def get_rf_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

