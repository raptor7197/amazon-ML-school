import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import re
from pathlib import Path

# Define dataset folder
DATASET_FOLDER = Path('../dataset/')

def load_data():
    """Load train and test data"""
    train = pd.read_csv(DATASET_FOLDER / 'train.csv')
    test = pd.read_csv(DATASET_FOLDER / 'test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    return train, test

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

def create_baseline_model():
    """Create and train baseline TF-IDF + Ridge model"""
    # Load data
    train, test = load_data()

    # Parse numerical features
    print("Parsing numerical features...")
    train_parsed = train['catalog_content'].apply(parse_catalog_content)
    train = train.join(pd.DataFrame(list(train_parsed)))

    test_parsed = test['catalog_content'].apply(parse_catalog_content)
    test = test.join(pd.DataFrame(list(test_parsed)))

    # Fill missing values
    train['value'] = train['value'].fillna(0.0)
    test['value'] = test['value'].fillna(0.0)
    train['unit'] = train['unit'].fillna('unknown')
    test['unit'] = test['unit'].fillna('unknown')

    # Preprocess text
    print("Preprocessing text...")
    train['processed_text'] = train['catalog_content'].apply(preprocess_text)
    test['processed_text'] = test['catalog_content'].apply(preprocess_text)

    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_text = tfidf.fit_transform(train['processed_text'])
    X_test_text = tfidf.transform(test['processed_text'])

    # Encode units
    print("Encoding categorical features...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    unit_train = encoder.fit_transform(train[['unit']])
    unit_test = encoder.transform(test[['unit']])

    # Numerical features
    value_train = train[['value']].values
    value_test = test[['value']].values

    # Combine features
    from scipy.sparse import hstack
    X_train = hstack([X_train_text, unit_train, value_train])
    X_test = hstack([X_test_text, unit_test, value_test])

    y_train = train['price'].values

    print(f"Combined features shape: {X_train.shape}")

    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train Ridge model
    print("Training Ridge model...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_split, y_train_split)

    # Validate
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(".4f")

    # Train on full data
    print("Training on full dataset...")
    model.fit(X_train, y_train)

    # Predict on test
    print("Generating predictions...")
    test_predictions = model.predict(X_test)

    # Ensure non-negative predictions
    test_predictions = np.maximum(test_predictions, 0)

    return test['sample_id'].values, test_predictions

def create_submission(sample_ids, predictions, filename='submission.csv'):
    """Create submission CSV"""
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })

    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"Submission shape: {submission.shape}")
    print(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")

if __name__ == "__main__":
    # Create baseline model and get predictions
    sample_ids, predictions = create_baseline_model()

    # Create submission
    create_submission(sample_ids, predictions)