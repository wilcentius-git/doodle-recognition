from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_rf_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

def get_svm_model():
    return SVC(
        probability=True, 
        kernel='linear',  # Linear bagus untuk high-dimensional
        C=10,
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )