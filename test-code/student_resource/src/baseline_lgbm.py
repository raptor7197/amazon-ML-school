
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import re
from pathlib import Path

# Define dataset folder
DATASET_FOLDER = Path('student_resource/dataset/')

def load_data():
    """Load train and test data"""
    train = pd.read_csv(DATASET_FOLDER / 'train.csv')
    test = pd.read_csv(DATASET_FOLDER / 'test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    return train, test

def preprocess_text(text):
    """Basic text preprocessing for TF-IDF"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def parse_catalog_content(content):
    """Parse catalog_content to extract numerical features"""
    if pd.isna(content):
        return {'value': 0.0, 'unit': 'unknown'}

    lines = content.split('\n')
    value = 0.0
    unit = 'unknown'

    for line in lines:
        line = line.strip()
        if line.startswith('Value:'):
            value_str = line.replace('Value:', '').strip()
            try:
                value = float(value_str)
            except ValueError:
                value = 0.0
        elif line.startswith('Unit:'):
            unit = line.replace('Unit:', '').strip().lower()

    return {'value': value, 'unit': unit}

def create_baseline_model():
    """Create and train baseline TF-IDF + LightGBM model"""
    # Load data
    train, test = load_data()

    # Preprocess text
    print("Preprocessing text...")
    train['processed_text'] = train['catalog_content'].apply(preprocess_text)
    test['processed_text'] = test['catalog_content'].apply(preprocess_text)

    # Parse IPQ
    print("Parsing IPQ...")
    train_parsed = train['catalog_content'].apply(parse_catalog_content)
    train = train.join(pd.DataFrame(list(train_parsed)))
    test_parsed = test['catalog_content'].apply(parse_catalog_content)
    test = test.join(pd.DataFrame(list(test_parsed)))
    train['value'] = train['value'].fillna(0.0)
    test['value'] = test['value'].fillna(0.0)

    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_text = tfidf.fit_transform(train['processed_text'])
    X_test_text = tfidf.transform(test['processed_text'])

    # Combine features
    from scipy.sparse import hstack
    X_train = hstack([X_train_text, train[['value']].values]).tocsr()
    X_test = hstack([X_test_text, test[['value']].values]).tocsr()

    y_train = train['price'].values

    # 5-fold CV
    print("Performing 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Fold {fold+1}...")
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        lgbm = LGBMRegressor(random_state=42)
        lgbm.fit(X_train_fold, y_train_fold)
        
        oof_preds[val_index] = lgbm.predict(X_val_fold)
        test_preds += lgbm.predict(X_test) / 5

    rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"5-fold CV RMSE: {rmse:.4f}")

    # Simple mean predictor
    mean_pred = np.mean(y_train)
    mean_rmse = np.sqrt(mean_squared_error(y_train, np.full_like(y_train, mean_pred)))
    print(f"Simple mean predictor RMSE: {mean_rmse:.4f}")

    # Create submission file
    submission = pd.DataFrame({'sample_id': test['sample_id'], 'price': test_preds})
    submission.to_csv('submission_baseline.csv', index=False)
    print("Submission file created: submission_baseline.csv")

if __name__ == "__main__":
    create_baseline_model()
