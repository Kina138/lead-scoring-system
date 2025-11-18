# 🚀 Complete Setup Guide for GitHub Codespaces

## BƯỚC 1: Tạo GitHub Codespace

1. Tạo repository mới trên GitHub
2. Upload toàn bộ folder `lead-scoring-system` lên repository
3. Click nút "Code" → "Codespaces" → "Create codespace on main"
4. Đợi Codespace khởi động (khoảng 2-3 phút)

## BƯỚC 2: Install Dependencies

```bash
# Mở terminal trong Codespace và chạy:
pip install -r requirements.txt
```

**Lưu ý:** Quá trình install có thể mất 5-10 phút. Hãy kiên nhẫn!

## BƯỚC 3: Download Dataset

### Option A: Manual Download (Khuyến nghị cho người mới)

1. Truy cập: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset
2. Download file `Leads.csv`
3. Upload vào thư mục `data/raw/` trong Codespace

### Option B: Kaggle API (Nâng cao)

```bash
# Install Kaggle
pip install kaggle

# Setup Kaggle credentials (cần API token từ Kaggle.com)
mkdir -p ~/.kaggle
# Copy your kaggle.json vào ~/.kaggle/

# Download dataset
cd data/raw
kaggle datasets download -d amritachatterjee09/lead-scoring-dataset
unzip lead-scoring-dataset.zip
rm lead-scoring-dataset.zip
cd ../..
```

## BƯỚC 4: Verify Setup

```bash
# Kiểm tra dataset đã có chưa
ls -lh data/raw/Leads.csv

# Nếu thấy file (~2MB) là OK!
```

## BƯỚC 5: Run Data Pipeline

### 5.1. Test Data Loading

```bash
python src/data/data_loader.py
```

**Expected Output:**
```
============================================================
                       Loading Dataset
============================================================
✓ Loaded 9,240 records from data/raw/Leads.csv
  Columns: 37
```

### 5.2. Test Preprocessing

```bash
python src/data/preprocessor.py
```

**Expected Output:**
```
✓ Preprocessing complete
  Features: 30+
  Samples: 9240
```

## BƯỚC 6: Train All Models

```bash
python src/models/model_trainer.py
```

**Thời gian:** ~10-15 phút (tùy CPU của Codespace)

**Expected Output:**
```
============================================================
Training Logistic Regression...
✓ Logistic Regression training complete
Accuracy: 0.8234
AUC: 0.8756
...
🏆 Best Model: Gradient Boosting
```

**Models sẽ được lưu tại:** `models/*.pkl`

## BƯỚC 7: Start Web Application

```bash
cd web
python app.py
```

**Expected Output:**
```
✓ Models loaded successfully
 * Running on http://0.0.0.0:5000
```

### Access Web App:

1. Codespace sẽ tự động forward port 5000
2. Click notification "Open in Browser" HOẶC
3. Go to "Ports" tab → Click địa chỉ port 5000

## BƯỚC 8: Test Web Interface

1. Click "Upload Data" trên navbar
2. Upload file `Leads.csv` (từ data/raw/)
3. Xem predictions và recommendations
4. Download results CSV

## 🎯 Testing Individual Components

### Test Models Individually:

```bash
# Logistic Regression
python src/models/logistic_model.py

# Random Forest
python src/models/random_forest_model.py

# Gradient Boosting
python src/models/gradient_boosting_model.py

# Neural Network
python src/models/neural_network_model.py
```

### Test SHAP Explainability:

```python
# Create test script
python << 'EOF'
import sys
sys.path.append('.')
from src.explainability.shap_explainer import SHAPExplainer
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
import joblib

# Load data
loader = DataLoader()
df = loader.load()

preprocessor = DataPreprocessor()
df_processed = preprocessor.fit_transform(df)
X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)

# Load best model
model = joblib.load('models/gradient_boosting.pkl')

# SHAP analysis
explainer = SHAPExplainer(model, X_train[:100], model_type='tree')
explainer.plot_summary(X_test[:200], X_test.columns)
print("✓ SHAP plots saved to outputs/visualizations/")
EOF
```

### Test Recommendations Generator:

```python
python << 'EOF'
import sys
sys.path.append('.')
from src.generative.template_generator import TemplateGenerator
import pandas as pd

generator = TemplateGenerator()

# Test lead
test_lead = {
    'name': 'John Doe',
    'Specialization': 'IT',
    'Total Time Spent on Website': 1200,
    'conversion_probability': 0.85
}

rec = generator.generate_recommendation(test_lead, 'High')
print("="*60)
print(f"Subject: {rec['email_subject']}")
print("\nMessage:")
print(rec['message'])
print("\nChannel:", rec['channel'])
print("Priority:", rec['priority'])
EOF
```

## 📊 Expected Results

After completing all steps, you should have:

✅ **Trained Models** (in `models/` directory):
- logistic_regression.pkl
- random_forest.pkl
- gradient_boosting.pkl (~best model)
- neural_network.h5
- preprocessor.pkl

✅ **Performance Metrics:**
- Logistic Regression: AUC ~0.87
- Random Forest: AUC ~0.91
- **Gradient Boosting: AUC ~0.93** 🏆
- Neural Network: AUC ~0.92

✅ **Web Application:**
- Running on port 5000
- Can upload CSV
- View predictions
- Download results

✅ **Visualizations** (in `outputs/visualizations/`):
- model_comparison.png
- shap_summary.png
- shap_feature_importance.png

## 🐛 Troubleshooting

### Issue: "File not found: Leads.csv"
**Solution:** Download dataset từ Kaggle và đặt vào `data/raw/`

### Issue: "Module not found"
**Solution:** 
```bash
pip install -r requirements.txt
# Hoặc install từng package:
pip install pandas scikit-learn tensorflow shap flask
```

### Issue: "Model not loaded" khi chạy web app
**Solution:** Train models trước:
```bash
python src/models/model_trainer.py
```

### Issue: Port 5000 already in use
**Solution:**
```bash
# Change port in web/app.py (line cuối):
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: Out of memory khi training Neural Network
**Solution:** Giảm batch_size trong `configs/config.yaml`:
```yaml
neural_network:
  batch_size: 16  # thay vì 32
```

## 📝 Next Steps

1. **Experiment với hyperparameters** trong `configs/config.yaml`
2. **Add more features** trong `preprocessor.py`
3. **Customize templates** trong `template_generator.py`
4. **Add more visualizations** trong notebooks
5. **Deploy to production** (Heroku, AWS, etc.)

## 🎓 For Capstone Presentation

### Files quan trọng để demo:

1. **README.md** - Overview
2. **src/models/model_trainer.py** - Training pipeline
3. **web/app.py** - Web application
4. **outputs/visualizations/** - Plots for slides
5. **outputs/reports/model_comparison.csv** - Results table

### Demo Flow:

1. Show codebase structure
2. Run data loader → Show EDA
3. Run model trainer → Show comparison
4. Show SHAP visualizations
5. Demo web application
6. Show recommendations output

## 📞 Support

Nếu gặp vấn đề:
1. Check error message carefully
2. Verify all files exist
3. Check Python version (should be 3.9+)
4. Try reinstalling dependencies

**Contact:** buianhthivan@cityuniversity.edu

Good luck với capstone! 🚀
