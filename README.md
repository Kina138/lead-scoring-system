# 🎯 AI-Based Lead Scoring and Generative Marketing Recommendation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.13.0-orange.svg)](https://www.tensorflow.org/)

An end-to-end AI system that predicts lead conversion probability, explains predictions using SHAP, and generates personalized marketing recommendations.

## 📋 Features

- ✅ **Predictive Analytics**: 4 ML models (Logistic Regression, Random Forest, Gradient Boosting, Neural Network)
- ✅ **Explainable AI**: SHAP visualizations for model interpretability
- ✅ **Generative AI**: Automated marketing recommendation generation
- ✅ **Web Interface**: User-friendly Flask app for non-technical users
- ✅ **Lead Segmentation**: Automatic High/Medium/Low priority classification

## 🚀 Quick Start on GitHub Codespaces

### Step 1: Open in Codespaces

Click the green "Code" button → "Codespaces" → "Create codespace on main"

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

```bash
# Option A: Manual download
# Go to https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset
# Download and place in data/raw/

# Option B: Using Kaggle API (requires setup)
cd data/raw
kaggle datasets download -d amritachatterjee09/lead-scoring-dataset
unzip lead-scoring-dataset.zip
cd ../..
```

### Step 4: Run Data Pipeline

```bash
# Execute notebooks in order
jupyter nbconvert --execute notebooks/01_data_exploration.ipynb
jupyter nbconvert --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --execute notebooks/03_model_training.ipynb
```

### Step 5: Start Web Application

```bash
cd web
python app.py
```

Access at: `http://localhost:5000`

## 📁 Project Structure

```
lead-scoring-system/
├── data/                          # Data storage
│   ├── raw/                       # Original dataset
│   ├── processed/                 # Cleaned data
│   └── uploads/                   # User uploads
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/                           # Source code
│   ├── data/                      # Data processing
│   ├── models/                    # ML models
│   ├── explainability/            # SHAP analysis
│   ├── generative/                # AI recommendations
│   └── utils/                     # Helper functions
├── web/                           # Flask application
│   ├── app.py                     # Main Flask app
│   ├── templates/                 # HTML templates
│   └── static/                    # CSS, JS, images
├── models/                        # Saved trained models
├── outputs/                       # Results and visualizations
└── tests/                         # Unit tests
```

## 🔧 Usage

### Command Line Interface

```python
# Train all models
python src/models/model_trainer.py

# Make predictions on new data
python src/models/predict.py --input data/uploads/new_leads.csv --output outputs/predictions/

# Generate SHAP explanations
python src/explainability/shap_explainer.py --model models/gradient_boosting.pkl
```

### Web Interface

1. Upload CSV file with lead data
2. View predictions and segmentation
3. Explore SHAP explanations
4. Download marketing recommendations

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.82 | 0.78 | 0.75 | 0.87 |
| Random Forest | 0.87 | 0.84 | 0.81 | 0.91 |
| **Gradient Boosting** | **0.89** | **0.86** | **0.84** | **0.93** |
| Neural Network | 0.88 | 0.85 | 0.82 | 0.92 |

*Note: These are expected performance metrics. Actual results may vary.*

## 🎓 Dataset

**Source**: [X Education Lead Scoring Dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)

- **Size**: 9,240 leads
- **Features**: 37 (demographic, behavioral, engagement)
- **Target**: Binary conversion (0/1)
- **Class Distribution**: 61.5% not converted, 38.5% converted

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📝 Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  test_size: 0.2
  random_state: 42
  
preprocessing:
  handle_missing: median
  scale_features: true
  
segmentation:
  high_threshold: 0.7
  low_threshold: 0.3
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes (CS 687 Capstone Project).

## 👤 Author

**Anh Thi Van Bui**
- Advisor: Sivakumar Visweswaran
- City University of Seattle
- Fall 2025

## 🙏 Acknowledgments

- X Education for the dataset
- City University of Seattle
- Research papers cited in the project documentation

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact: buianhthivan@cityuniversity.edu
