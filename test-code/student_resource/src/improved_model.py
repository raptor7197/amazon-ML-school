import os
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import optuna
import shap
import matplotlib.pyplot as plt

# Image processing imports
import timm
import torch
from torchvision import transforms
from PIL import Image
from utils import download_images

# Define paths
DATASET_FOLDER = Path('student_resource/dataset/')

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

def load_data(sample_size=None):
    """Load train and test data, optionally sample for faster testing"""
    train = pd.read_csv(DATASET_FOLDER / 'train.csv')
    test = pd.read_csv(DATASET_FOLDER / 'test.csv')

    if sample_size:
        train = train.sample(sample_size, random_state=42).reset_index(drop=True)
        test = test.sample(min(sample_size, len(test)), random_state=42).reset_index(drop=True)

    # Parse numerical features
    train_parsed = train['catalog_content'].apply(parse_catalog_content)
    train = train.join(pd.DataFrame(list(train_parsed)))

    test_parsed = test['catalog_content'].apply(parse_catalog_content)
    test = test.join(pd.DataFrame(list(test_parsed)))

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    return train, test

def load_text_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

def extract_text_features(texts, model):
    """Extract text embeddings"""
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def load_image_model():
    """Load ResNet18 model for feature extraction"""
    model = timm.create_model('resnet18', pretrained=True, num_classes=0)  # Remove classifier
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for ResNet18"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # Add batch dim
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_image_features(image_links, download_folder='images'):
    """Download images and extract features"""
    print("Downloading images...")
    download_images(image_links, download_folder)

    print("Loading image model...")
    model = load_image_model()

    features = []
    for link in image_links:
        filename = Path(link).name
        image_path = os.path.join(download_folder, filename)
        if os.path.exists(image_path):
            tensor = preprocess_image(image_path)
            if tensor is not None:
                with torch.no_grad():
                    feat = model(tensor).squeeze().numpy()
                features.append(feat)
            else:
                features.append(np.zeros(512))  # ResNet18 features are 512d
        else:
            features.append(np.zeros(512))

    return np.array(features)

def create_multimodal_features(train, test, sample_size=None):
    """Create combined text and image features"""
    # Load models
    print("Loading models...")
    text_model = load_text_model()

    # Extract text features
    print("Extracting text features...")
    train_text_features = extract_text_features(train['catalog_content'].tolist(), text_model)
    test_text_features = extract_text_features(test['catalog_content'].tolist(), text_model)

    # Extract image features
    print("Extracting image features...")
    download_folder = 'images'
    train_image_features = extract_image_features(train['image_link'].tolist(), download_folder)
    test_image_features = extract_image_features(test['image_link'].tolist(), download_folder)

    # Combine features
    print("Combining features...")
    X_train = np.hstack([train_text_features, train_image_features])
    X_test = np.hstack([test_text_features, test_image_features])

    y_train = train['price'].values

    print(f"Combined features shape: {X_train.shape}")

    feature_names = [f'text_feature_{i}' for i in range(train_text_features.shape[1])]
    feature_names += [f'image_feature_{i}' for i in range(train_image_features.shape[1])]

    return X_train, X_test, y_train, feature_names

def train_improved_model(X_train, y_train):
    """Train Ridge model on multimodal features"""
    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Tune LGBM hyperparameters
    print("Tuning LGBM hyperparameters...")
    best_params = tune_lgbm(X_train, y_train)

    # Train ensemble model
    print("Training ensemble model (Ridge + LGBM) on multimodal features...")
    ridge = Ridge(alpha=1.0)
    lgbm = LGBMRegressor(**best_params, random_state=42)
    model = VotingRegressor([('ridge', ridge), ('lgbm', lgbm)])
    model.fit(X_train_split, y_train_split)

    # Validate
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE: {val_rmse:.4f}")

    # Train on full data
    model.fit(X_train, y_train)

    return model

def tune_lgbm(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }
        
        model = LGBMRegressor(**params, random_state=42)
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        model.fit(X_train_split, y_train_split)
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    print('Best trial:')
    trial = study.best_trial
    
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        
    return study.best_params

def create_submission(sample_ids, predictions, filename='improved_submission.csv'):
    """Create submission CSV"""
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })

    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"Submission shape: {submission.shape}")
    print(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")

def explain_model(model, X_train, feature_names):
    print("Explaining model with SHAP...")
    explainer = shap.TreeExplainer(model.named_estimators_['lgbm'])
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig('shap_summary_plot.png')
    plt.close()

def main(sample_size=100):  # Use small sample for demo
    """Main function to run improved multimodal model"""
    print("Starting improved multimodal model...")

    # Load data
    train, test = load_data(sample_size=sample_size)

    # Create multimodal features
    X_train, X_test, y_train, feature_names = create_multimodal_features(train, test, sample_size)

    # Train model
    model = train_improved_model(X_train, y_train)

    # Predict
    print("Generating predictions...")
    test_predictions = model.predict(X_test)
    test_predictions = np.maximum(test_predictions, 0)  # Non-negative

    # Create submission
    create_submission(test['sample_id'].values, test_predictions, 'improved_submission.csv')

    # Explain model
    explain_model(model, X_train, feature_names)

if __name__ == "__main__":
    main()