#!/bin/bash

# Quick Start Script for AI Lead Scoring System
# Author: Anh Thi Van Bui

echo "============================================================"
echo "  AI Lead Scoring System - Quick Start"
echo "  CS 687 Capstone Project"
echo "============================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 ]] || [[ $(echo "$python_version" | cut -d. -f2) -lt 9 ]]; then
    print_error "Python 3.9+ required. Current version: $python_version"
    exit 1
fi
print_success "Python version OK"
echo ""

# Create .gitkeep files for empty directories
echo "Setting up directory structure..."
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/uploads/.gitkeep
touch models/.gitkeep
touch outputs/predictions/.gitkeep
touch outputs/reports/.gitkeep
touch outputs/visualizations/.gitkeep
print_success "Directory structure ready"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "(This may take 5-10 minutes)"
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi
echo ""

# Check for dataset
echo "Checking for dataset..."
if [ -f "data/raw/Leads.csv" ]; then
    print_success "Dataset found!"
    file_size=$(du -h data/raw/Leads.csv | cut -f1)
    echo "  File size: $file_size"
else
    print_warning "Dataset NOT found"
    echo ""
    echo "Please download dataset from:"
    echo "https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset"
    echo ""
    echo "And place Leads.csv in: data/raw/"
    echo ""
    read -p "Press Enter after downloading dataset..."
fi
echo ""

# Test data loading
echo "Testing data loader..."
python3 src/data/data_loader.py > /tmp/loader_test.log 2>&1

if [ $? -eq 0 ]; then
    print_success "Data loader working"
else
    print_warning "Data loader test failed (dataset might not be present)"
fi
echo ""

# Menu
echo "============================================================"
echo "  What would you like to do?"
echo "============================================================"
echo "1. Train all models (10-15 minutes)"
echo "2. Start web application"
echo "3. Run Jupyter notebook"
echo "4. Test individual components"
echo "5. View setup guide"
echo "6. Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Starting model training pipeline..."
        echo "This will take 10-15 minutes depending on your CPU"
        python3 src/models/model_trainer.py
        
        if [ $? -eq 0 ]; then
            print_success "Training complete!"
            echo ""
            echo "Models saved in: models/"
            echo "Results saved in: outputs/"
        else
            print_error "Training failed. Check errors above."
        fi
        ;;
    
    2)
        echo ""
        if [ ! -f "models/gradient_boosting.pkl" ]; then
            print_warning "No trained models found!"
            echo "Please train models first (option 1)"
            exit 1
        fi
        
        echo "Starting web application..."
        echo "Access at: http://localhost:5000"
        echo "Press Ctrl+C to stop"
        cd web
        python3 app.py
        ;;
    
    3)
        echo ""
        echo "Starting Jupyter notebook..."
        jupyter notebook notebooks/
        ;;
    
    4)
        echo ""
        echo "============================================================"
        echo "  Test Menu"
        echo "============================================================"
        echo "1. Test data preprocessing"
        echo "2. Test logistic regression"
        echo "3. Test random forest"
        echo "4. Test gradient boosting"
        echo "5. Test neural network"
        echo "6. Test SHAP explainer"
        echo "7. Test template generator"
        echo ""
        read -p "Enter choice [1-7]: " test_choice
        
        case $test_choice in
            1) python3 src/data/preprocessor.py ;;
            2) python3 src/models/logistic_model.py ;;
            3) python3 src/models/random_forest_model.py ;;
            4) python3 src/models/gradient_boosting_model.py ;;
            5) python3 src/models/neural_network_model.py ;;
            6) 
                if [ -f "models/gradient_boosting.pkl" ]; then
                    python3 -c "
import sys
sys.path.append('.')
from src.explainability.shap_explainer import SHAPExplainer
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
import joblib

loader = DataLoader()
df = loader.load()
preprocessor = DataPreprocessor()
df_processed = preprocessor.fit_transform(df)
X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)

model = joblib.load('models/gradient_boosting.pkl')
explainer = SHAPExplainer(model, X_train[:100], model_type='tree')
explainer.plot_summary(X_test[:200], X_test.columns)
print('✓ SHAP visualizations saved')
"
                else
                    print_error "Please train models first"
                fi
                ;;
            7)
                python3 -c "
import sys
sys.path.append('.')
from src.generative.template_generator import TemplateGenerator

generator = TemplateGenerator()
test_lead = {
    'name': 'Test Lead',
    'Specialization': 'IT',
    'Total Time Spent on Website': 1200,
    'conversion_probability': 0.85
}
rec = generator.generate_recommendation(test_lead, 'High')
print('='*60)
print(f\"Subject: {rec['email_subject']}\")
print(f\"\nMessage:\n{rec['message']}\")
print(f\"\nChannel: {rec['channel']}\")
print(f\"Priority: {rec['priority']}\")
print('='*60)
"
                ;;
        esac
        ;;
    
    5)
        echo ""
        if command -v less &> /dev/null; then
            less SETUP_GUIDE.md
        else
            cat SETUP_GUIDE.md
        fi
        ;;
    
    6)
        echo "Goodbye!"
        exit 0
        ;;
    
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
print_success "Done!"
echo "============================================================"
