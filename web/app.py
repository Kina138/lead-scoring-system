"""
Flask web application for AI Lead Scoring System with SHAP Integration
"""
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import pandas as pd
import os
import joblib
from werkzeug.utils import secure_filename
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import shap
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.generative.template_generator import TemplateGenerator
from src.utils.helpers import classify_segment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = '../data/uploads'
app.config['OUTPUT_FOLDER'] = '../outputs/predictions'
app.config['SHAP_FOLDER'] = 'static/shap_plots'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure all required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['SHAP_FOLDER'], exist_ok=True)

# Global variables for models
model = None
preprocessor = None
generator = None
shap_explainer = None
feature_names = None

def load_models():
    """Load trained models and initialize SHAP explainer"""
    global model, preprocessor, generator, shap_explainer, feature_names
    
    try:
        # Load model and preprocessor
        model = joblib.load('../models/gradient_boosting.pkl')
        preprocessor = DataPreprocessor()
        preprocessor.load('../models/preprocessor.pkl')
        generator = TemplateGenerator()
        
        # Get feature names from preprocessor
        feature_names = preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else None
        
        # Initialize SHAP explainer with the model
        # Use a small background dataset for faster computation
        try:
            # Load a sample of training data for SHAP background
            sample_data = pd.read_csv('../data/raw/Leads.csv').head(100)
            if 'Converted' in sample_data.columns:
                sample_data = sample_data.drop('Converted', axis=1)
            X_background = preprocessor.transform(sample_data)
            
            # Create SHAP explainer
            shap_explainer = shap.TreeExplainer(model)
            print("✓ SHAP explainer initialized successfully")
        except Exception as e:
            print(f"⚠ SHAP explainer initialization failed: {e}")
            shap_explainer = None
        
        print("✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠ Models not found: {e}")
        return False

# Load models on startup
load_models()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return redirect(url_for('predict', filename=filename))
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict(filename):
    """Make predictions on uploaded file with SHAP explanations"""
    if model is None:
        flash('Models not loaded. Please train models first.', 'error')
        return redirect(url_for('index'))
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        
        # Store original data for display
        df_original = df.copy()
        
        # Remove target column if present (for prediction)
        if 'Converted' in df.columns:
            df = df.drop('Converted', axis=1)
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        predictions = model.predict_proba(X)[:, 1]
        
        # Add results to dataframe
        df_original['conversion_probability'] = predictions
        df_original['segment'] = df_original['conversion_probability'].apply(classify_segment)
        
        # Generate AI recommendations
        print("Generating AI recommendations...")
        recommendations_df = generator.generate_batch_recommendations(df_original)
        
        # Combine results
        results_df = pd.concat([df_original, recommendations_df], axis=1)
        
        # Save results
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'results_{filename}')
        results_df.to_csv(output_path, index=False)
        
        # Generate SHAP visualizations
        shap_plots = {}
        if shap_explainer is not None:
            print("Generating SHAP explanations...")
            try:
                shap_plots = generate_shap_visualizations(X, predictions, filename)
            except Exception as e:
                print(f"⚠ SHAP visualization failed: {e}")
                shap_plots = {}
        
        # Calculate statistics
        stats = {
            'total_leads': len(df_original),
            'high_priority': len(df_original[df_original['segment'] == 'High']),
            'medium_priority': len(df_original[df_original['segment'] == 'Medium']),
            'low_priority': len(df_original[df_original['segment'] == 'Low']),
            'avg_score': float(predictions.mean()),
            'filename': f'results_{filename}'
        }
        
        # Get top leads for display
        top_leads = results_df.nlargest(50, 'conversion_probability').to_dict('records')
        
        # Get feature importance from SHAP
        feature_importance = None
        if shap_explainer is not None and feature_names is not None:
            try:
                shap_values = shap_explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                
                importance = np.abs(shap_values).mean(axis=0)
                feature_importance = [
                    {'feature': name, 'importance': float(imp)}
                    for name, imp in zip(feature_names, importance)
                ]
                feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
            except Exception as e:
                print(f"⚠ Feature importance calculation failed: {e}")
        
        return render_template('results.html', 
                             leads=top_leads, 
                             stats=stats,
                             shap_plots=shap_plots,
                             feature_importance=feature_importance)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error details:\n{error_detail}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))

def generate_shap_visualizations(X, predictions, filename):
    """Generate SHAP visualizations and return paths"""
    shap_plots = {}
    
    try:
        # Calculate SHAP values
        shap_values = shap_explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        # 1. Summary Plot (Bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, 
                         feature_names=feature_names,
                         plot_type="bar",
                         max_display=15,
                         show=False)
        summary_bar_path = f'shap_plots/summary_bar_{filename}.png'
        plt.tight_layout()
        plt.savefig(os.path.join('static', summary_bar_path), dpi=150, bbox_inches='tight')
        plt.close()
        shap_plots['summary_bar'] = summary_bar_path
        
        # 2. Summary Plot (Beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X,
                         feature_names=feature_names,
                         max_display=15,
                         show=False)
        summary_path = f'shap_plots/summary_{filename}.png'
        plt.tight_layout()
        plt.savefig(os.path.join('static', summary_path), dpi=150, bbox_inches='tight')
        plt.close()
        shap_plots['summary'] = summary_path
        
        # 3. Waterfall plot for top prediction
        top_idx = predictions.argmax()
        plt.figure(figsize=(10, 8))
        
        explanation = shap.Explanation(
            values=shap_values[top_idx],
            base_values=shap_explainer.expected_value if hasattr(shap_explainer, 'expected_value') else 0,
            data=X[top_idx],
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, max_display=15, show=False)
        waterfall_path = f'shap_plots/waterfall_top_{filename}.png'
        plt.tight_layout()
        plt.savefig(os.path.join('static', waterfall_path), dpi=150, bbox_inches='tight')
        plt.close()
        shap_plots['waterfall_top'] = waterfall_path
        
        # 4. Force plot for top prediction (as matplotlib)
        try:
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                shap_explainer.expected_value if hasattr(shap_explainer, 'expected_value') else 0,
                shap_values[top_idx],
                X[top_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            force_path = f'shap_plots/force_top_{filename}.png'
            plt.tight_layout()
            plt.savefig(os.path.join('static', force_path), dpi=150, bbox_inches='tight')
            plt.close()
            shap_plots['force_top'] = force_path
        except:
            pass  # Force plot might fail in some cases
        
        print(f"✓ Generated {len(shap_plots)} SHAP visualizations")
        
    except Exception as e:
        print(f"⚠ SHAP visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return shap_plots

@app.route('/download/<filename>')
def download(filename):
    """Download results file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

@app.route('/explain/<int:lead_id>')
def explain_lead(lead_id):
    """Generate individual SHAP explanation for a specific lead"""
    # This would be for interactive per-lead explanation
    # Implementation depends on how you want to store/retrieve individual predictions
    return jsonify({'message': 'Individual lead explanation feature coming soon'})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)