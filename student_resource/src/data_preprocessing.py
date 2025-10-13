import os
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Define dataset folder
DATASET_FOLDER = Path('../dataset/')

def load_data():
    """Load all CSV files into DataFrames"""
    train = pd.read_csv(DATASET_FOLDER / 'train.csv')
    test = pd.read_csv(DATASET_FOLDER / 'test.csv')
    sample_test = pd.read_csv(DATASET_FOLDER / 'sample_test.csv')
    sample_test_out = pd.read_csv(DATASET_FOLDER / 'sample_test_out.csv')

    print("Data loaded successfully!")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample test shape: {sample_test.shape}")
    print(f"Sample test out shape: {sample_test_out.shape}")

    return train, test, sample_test, sample_test_out

def explore_data(train, test, sample_test, sample_test_out):
    """Explore data structure and basic statistics"""
    print("\n=== DATA EXPLORATION ===")

    # Data types
    print("\nTrain dtypes:")
    print(train.dtypes)

    print("\nTest dtypes:")
    print(test.dtypes)

    # Missing values
    print("\nMissing values in train:")
    print(train.isnull().sum())

    print("\nMissing values in test:")
    print(test.isnull().sum())

    # Basic statistics for price
    print("\nPrice statistics in train:")
    print(train['price'].describe())

    # Sample data
    print("\nSample train data:")
    print(train.head())

    print("\nSample test data:")
    print(test.head())

def parse_catalog_content(content):
    """Parse catalog_content into structured fields"""
    if pd.isna(content):
        return {
            'item_name': None,
            'bullet_points': [],
            'product_description': None,
            'value': None,
            'unit': None
        }

    lines = content.split('\n')
    item_name = None
    bullet_points = []
    product_description = None
    value = None
    unit = None

    for line in lines:
        line = line.strip()
        if line.startswith('Item Name:'):
            item_name = line.replace('Item Name:', '').strip()
        elif line.startswith('Bullet Point'):
            bullet_points.append(line.split(':', 1)[1].strip() if ':' in line else line)
        elif line.startswith('Product Description:'):
            product_description = line.replace('Product Description:', '').strip()
        elif line.startswith('Value:'):
            value_str = line.replace('Value:', '').strip()
            try:
                value = float(value_str)
            except ValueError:
                value = None
        elif line.startswith('Unit:'):
            unit = line.replace('Unit:', '').strip()

    return {
        'item_name': item_name,
        'bullet_points': bullet_points,
        'product_description': product_description,
        'value': value,
        'unit': unit
    }

def preprocess_data(train, test):
    """Preprocess the data: parse catalog content, handle missing values"""
    print("\n=== DATA PREPROCESSING ===")

    # Parse catalog_content for train
    print("Parsing catalog content for train...")
    train_parsed = train['catalog_content'].apply(parse_catalog_content)
    train = train.join(pd.DataFrame(list(train_parsed)))

    # Parse catalog_content for test
    print("Parsing catalog content for test...")
    test_parsed = test['catalog_content'].apply(parse_catalog_content)
    test = test.join(pd.DataFrame(list(test_parsed)))

    # Handle missing values
    print("Handling missing values...")

    # For price in train, fill with median
    if 'price' in train.columns:
        train['price'] = train['price'].fillna(train['price'].median())

    # For text fields, fill with empty strings
    text_cols = ['catalog_content', 'item_name', 'product_description']
    for col in text_cols:
        if col in train.columns:
            train[col] = train[col].fillna('')
        if col in test.columns:
            test[col] = test[col].fillna('')

    # For value, fill with median
    if 'value' in train.columns:
        train['value'] = train['value'].fillna(train['value'].median())
    if 'value' in test.columns:
        test['value'] = test['value'].fillna(test['value'].median())

    # For unit, fill with 'Unknown'
    if 'unit' in train.columns:
        train['unit'] = train['unit'].fillna('Unknown')
    if 'unit' in test.columns:
        test['unit'] = test['unit'].fillna('Unknown')

    print("Preprocessing completed!")
    return train, test

def generate_summary_stats(train):
    """Generate summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")

    print(f"Total products: {len(train)}")
    print(f"Price range: ${train['price'].min():.2f} - ${train['price'].max():.2f}")
    print(f"Median price: ${train['price'].median():.2f}")
    print(f"Mean price: ${train['price'].mean():.2f}")

    # Top units
    print("\nTop 10 units:")
    print(train['unit'].value_counts().head(10))

    # Price distribution by unit
    print("\nMedian price by top units:")
    top_units = train['unit'].value_counts().head(5).index
    for unit in top_units:
        median_price = train[train['unit'] == unit]['price'].median()
        print(f"{unit}: ${median_price:.2f}")

if __name__ == "__main__":
    # Load data
    train, test, sample_test, sample_test_out = load_data()

    # Explore data
    explore_data(train, test, sample_test, sample_test_out)

    # Preprocess data
    train_processed, test_processed = preprocess_data(train.copy(), test.copy())

    # Generate summary stats
    generate_summary_stats(train_processed)

    print("\nData exploration and preprocessing completed!")