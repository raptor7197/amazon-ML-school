# Amazon Product Price Prediction

An advanced machine learning system for predicting product prices using catalog content (text) and product images.

## 📋 Overview

This project implements a comprehensive price prediction pipeline that combines:
- Text feature extraction using TF-IDF and transformer-based embeddings
- Image feature extraction using pre-trained CNNs (ResNet, CLIP)
- Ensemble modeling with LightGBM and neural networks
- Production-ready API deployment with FastAPI

## 🚀 Features

- **Multi-modal Learning**: Combines text and image features for robust predictions
- **Advanced NLP**: Leverages sentence-transformers and BERT embeddings
- **Computer Vision**: Uses state-of-the-art image models (CLIP, ResNet)
- **Ensemble Methods**: Combines multiple models for better performance
- **Production Ready**: Includes API, Docker support, and comprehensive testing
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Hyperparameter Optimization**: Automated tuning with Optuna

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amazon-price-predictor.git
cd amazon-price-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## 📁 Project Structure

```
amazon-price-predictor/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data directory (gitignored)
├── docs/                  # Documentation
├── experiments/           # Experiment logs and results
├── models/                # Saved models (gitignored)
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── api/              # FastAPI application
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model implementations
│   └── utils/            # Utility functions
├── tests/                 # Unit and integration tests
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project configuration
```

## 🏃 Quick Start

### 1. Data Preparation
```bash
python scripts/prepare_data.py --input student_resource/dataset/train.csv
```

### 2. Train Baseline Model
```bash
python scripts/train_baseline.py --config config/baseline.yaml
```

### 3. Train Advanced Model
```bash
python scripts/train_advanced.py --config config/advanced.yaml
```

### 4. Generate Predictions
```bash
python scripts/predict.py --model models/best_model.pkl --input student_resource/dataset/test.csv
```

### 5. Start API Server
```bash
uvicorn src.api.main:app --reload
```

## 📊 Model Pipeline

1. **Data Loading**: Efficient loading with pandas and data validation
2. **Feature Engineering**:
   - Text: TF-IDF, word embeddings, sentence embeddings
   - Images: Pre-trained CNN features (ResNet, CLIP)
   - Numerical: Price statistics, IPQ extraction
3. **Model Training**:
   - Baseline: TF-IDF + LightGBM
   - Advanced: Multi-modal embeddings + ensemble
4. **Evaluation**: K-fold cross-validation with comprehensive metrics
5. **Deployment**: FastAPI with async predictions

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters
- Feature engineering options
- Training settings
- API configuration

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## 🐳 Docker

Build the image:
```bash
docker build -t amazon-price-predictor .
```

Run the container:
```bash
docker run -p 8000:8000 amazon-price-predictor
```

## 📈 Performance

| Model | CV RMSE | Test RMSE | Training Time |
|-------|---------|-----------|---------------|
| Baseline (TF-IDF + LightGBM) | X.XX | X.XX | XX min |
| Advanced (Multi-modal) | X.XX | X.XX | XX min |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Amazon for the dataset
- Hugging Face for transformer models
- OpenAI for CLIP model