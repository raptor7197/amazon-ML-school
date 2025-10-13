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

def extract_specs(text):
    specs = {}
    
    # Storage (GB)
    storage_gb = re.findall(r'(\d+)\s?GB', text)
    if storage_gb:
        specs['storage_gb'] = int(storage_gb[0])
        
    # Size (ml, L)
    size_ml = re.findall(r'(\d+)\s?ml', text, re.IGNORECASE)
    if size_ml:
        specs['size_ml'] = int(size_ml[0])
        
    size_l = re.findall(r'(\d+)\s?L', text, re.IGNORECASE)
    if size_l:
        specs['size_l'] = int(size_l[0])
        
    # Dimensions (inch)
    dimensions_inch = re.findall(r'(\d+\.?\d*)\s?inch', text, re.IGNORECASE)
    if dimensions_inch:
        specs['dimensions_inch'] = float(dimensions_inch[0])
        
    # Battery (mAh)
    battery_mah = re.findall(r'(\d+)\s?mAh', text, re.IGNORECASE)
    if battery_mah:
        specs['battery_mah'] = int(battery_mah[0])
        
    return specs

def extract_category(text):
    text = text.lower()
    categories = {
        'electronics': ['electronics', 'camera', 'phone', 'headphone', 'laptop', 'computer'],
        'clothing': ['clothing', 'shirt', 'pant', 'dress', 'shoe'],
        'personal_care': ['personal care', 'shampoo', 'soap', 'lotion', 'cream'],
        'grocery': ['grocery', 'food', 'snack', 'drink'],
        'home': ['home', 'kitchen', 'furniture', 'decor'],
        'books': ['book', 'ebook', 'author'],
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return 'other'

def has_offer(text):
    text = text.lower()
    offer_patterns = ['mrp', '₹', 'rs.', 'discount']
    for pattern in offer_patterns:
        if pattern in text:
            return 1
    return 0

def count_adjectives(text):
    text = text.lower()
    
    positive_adjectives = ['premium', 'luxury', 'high-quality', 'excellent', 'great', 'durable', 'beautiful']
    negative_adjectives = ['cheap', 'low-quality', 'poor', 'bad', 'fake']
    
    adjective_counts = {'positive_adjectives': 0, 'negative_adjectives': 0}
    
    for adj in positive_adjectives:
        adjective_counts['positive_adjectives'] += text.count(adj)
        
    for adj in negative_adjectives:
        adjective_counts['negative_adjectives'] += text.count(adj)
        
    return adjective_counts

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

    # IPQ normalized features
    print("Creating IPQ normalized features...")
    train['price_per_unit'] = train['price'] / train['value']
    test['price_per_unit'] = test['price'] / test['value']

    # Fill inf/-inf with nan
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill nan with median
    train['price_per_unit'] = train['price_per_unit'].fillna(train['price_per_unit'].median())
    test['price_per_unit'] = test['price_per_unit'].fillna(test['price_per_unit'].median())

    # Brand extraction
    print("Extracting brand...")
    train['brand'] = train['item_name'].apply(lambda x: x.split(' ')[0])
    test['brand'] = test['item_name'].apply(lambda x: x.split(' ')[0])

    # Spec extraction
    print("Extracting specs...")
    train_specs = train['catalog_content'].apply(extract_specs)
    test_specs = test['catalog_content'].apply(extract_specs)

    train = train.join(pd.DataFrame(list(train_specs)))
    test = test.join(pd.DataFrame(list(test_specs)))

    spec_cols = ['storage_gb', 'size_ml', 'size_l', 'dimensions_inch', 'battery_mah']
    for col in spec_cols:
        if col in train.columns:
            train[col] = train[col].fillna(0)
        if col in test.columns:
            test[col] = test[col].fillna(0)

    # Category extraction
    print("Extracting category...")
    train['category'] = train['catalog_content'].apply(extract_category)
    test['category'] = test['catalog_content'].apply(extract_category)

    # Offer/MRP hints
    print("Extracting offer hints...")
    train['has_offer'] = train['catalog_content'].apply(has_offer)
    test['has_offer'] = test['catalog_content'].apply(has_offer)

    # Text sentiment / adjectives
    print("Extracting text sentiment...")
    train_adjectives = train['catalog_content'].apply(count_adjectives)
    test_adjectives = test['catalog_content'].apply(count_adjectives)

    train = train.join(pd.DataFrame(list(train_adjectives)))
    test = test.join(pd.DataFrame(list(test_adjectives)))

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