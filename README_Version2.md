# Amazon ML School - Product Pricing Challenge

## Team Details
- **Team Name:** TensorTitans
- **Team Members:** Rahul, Sachin, Vamsi, Atul

## Project Overview

This repository contains our solution for the Amazon ML School Product Pricing Challenge. Our approach combines multiple modeling techniques including deep learning, ensemble methods, and advanced feature engineering to predict optimal product prices.

## Repository Structure

### Core Files
- `TensorTitans.ipynb`: Main Jupyter notebook with complete solution pipeline
- `features.py`: Advanced feature extraction utilities for text processing
- `train_text_ensemble_elite.py`: Elite ensemble model with text embeddings and adversarial validation
- `TensorTitans.docx`: Detailed project documentation

### Baseline Models (samples/student_resource/src/)
- `baseline_model.py`: Basic neural network baseline
- `baseline_lgbm.py`: LightGBM baseline implementation
- `improved_model.py`: Enhanced deep learning model
- `data_preprocessing.py`: Data cleaning and preprocessing utilities
- `utils.py`: Helper functions and utilities
- `example.ipynb`: Tutorial notebook

## Methodology

### 1. Feature Engineering (`features.py`)

Our advanced feature extraction includes:
- **Text Processing**: Clean HTML tags, normalize whitespace
- **Brand Extraction**: Smart brand detection from product titles
- **Quantity Features**: Extract pack sizes, volumes, weights using regex patterns
- **Item Pack Quantity (IPQ)**: Specialized extraction for product quantities
- **Text Statistics**: Length, digit counts, and other linguistic features

Key innovations:
- Multi-scale quantity normalization (ml/l, g/kg, cm/m conversions)
- Logarithmic quantity bucketing for better distribution
- Stop-word filtering for brand detection

### 2. Elite Ensemble Model (`train_text_ensemble_elite.py`)

Our production model combines:
- **Text Embeddings**: TF-IDF (word + character n-grams) + SVD dimensionality reduction
- **Multiple Regressors**: Ridge, ElasticNet, HuberRegressor for robustness
- **Gradient Boosting**: LightGBM with quantile regression (0.3, 0.5, 0.7 quantiles)
- **XGBoost**: Tweedie regression for price distribution modeling
- **Adversarial Validation**: Domain adaptation weights for train/test distribution shift
- **Meta-Learning**: Ridge regression on stacked predictions
- **Post-Processing**: Brand/quantity-based calibration with shrinkage

### 3. Advanced Techniques

- **Cross-Validation**: Stratified 5-fold based on price quantiles
- **Ensemble Optimization**: Scipy optimization for optimal blending weights
- **Adversarial Weighting**: Logistic regression to detect domain shift
- **Calibration**: Group-wise median adjustments with confidence-based shrinkage

## Usage

### Quick Start
```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost scipy tqdm

# Run elite ensemble model
python train_text_ensemble_elite.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --out_dir outputs/ \
    --out_csv submission.csv \
    --folds 5
```

### Jupyter Notebook
Open `TensorTitans.ipynb` for interactive exploration and complete pipeline.

## Model Performance

Our ensemble approach achieves superior performance through:
- **Diversity**: Multiple algorithm types (linear, tree-based, neural)
- **Robustness**: Quantile regression and adversarial validation
- **Calibration**: Post-hoc adjustments based on product characteristics
- **Meta-Learning**: Second-level model to combine predictions optimally

## Technical Innovations

1. **Advanced Feature Engineering**: Comprehensive text parsing with domain-specific patterns
2. **Adversarial Validation**: Automatic detection and correction of train/test distribution shifts
3. **Multi-Quantile Ensemble**: Captures price uncertainty through quantile predictions
4. **Hierarchical Calibration**: Brand and quantity-aware post-processing
5. **Shrinkage Methodology**: Confidence-based adjustment factors

## Files and Dependencies

### Core Dependencies
- pandas, numpy: Data manipulation
- scikit-learn: ML algorithms and preprocessing
- lightgbm, xgboost: Gradient boosting
- scipy: Optimization and sparse matrices
- tqdm: Progress tracking

### Baseline Dependencies (for reference)
- tensorflow: Deep learning models
- sentence-transformers: Text embeddings
- pillow: Image processing
- matplotlib, seaborn: Visualization

## Results and Submission

The model outputs predictions in the required format:
- `sample_id`: Test sample identifier
- `price`: Predicted price (positive, rounded to 2 decimals)

All models use only provided data and open-source libraries (MIT/Apache 2.0 licensed).

## Future Work

- **Transformer Models**: BERT/RoBERTa for text understanding
- **Graph Neural Networks**: Product relationship modeling
- **Multi-Task Learning**: Joint price and category prediction
- **Automated Feature Engineering**: Neural architecture search for features

## Contact

For questions or collaboration: rahul.b2024@vitstudent.ac.in