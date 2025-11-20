"""
Flask web application for AI Lead Scoring System
"""
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import pandas as pd
import os
import joblib
from werkzeug.utils import secure_filename
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.generative.template_generator import TemplateGenerator
from src.utils.helpers import classify_segment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = '../data/uploads'
app.config['OUTPUT_FOLDER'] = '../outputs/predictions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure all required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load best model and preprocessor
try:
    model = joblib.load('../models/gradient_boosting.pkl')
    preprocessor = DataPreprocessor()
    preprocessor.load('../models/preprocessor.pkl')
    generator = TemplateGenerator()
    print("✓ Models loaded successfully")
except Exception as e:
    model, preprocessor, generator = None, None, None
    print(f"⚠ Models not found: {e}")

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
    """Make predictions on uploaded file"""
    if model is None:
        flash('Models not loaded. Please train models first.', 'error')
        return redirect(url_for('index'))
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)
        
        # Remove target column if present (for prediction)
        if 'Converted' in df.columns:
            df = df.drop('Converted', axis=1)
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        predictions = model.predict_proba(X)[:, 1]
        
        # Add results to dataframe
        df['conversion_probability'] = predictions
        df['segment'] = df['conversion_probability'].apply(classify_segment)
        
        # Generate AI recommendations
        print("Generating AI recommendations...")
        recommendations_df = generator.generate_batch_recommendations(df)
        
        # Combine results
        results_df = pd.concat([df, recommendations_df], axis=1)
        
        # Save results
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'results_{filename}')
        results_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        stats = {
            'total_leads': len(df),
            'high_priority': len(df[df['segment'] == 'High']),
            'medium_priority': len(df[df['segment'] == 'Medium']),
            'low_priority': len(df[df['segment'] == 'Low']),
            'avg_score': float(predictions.mean()),
            'filename': f'results_{filename}'
        }
        
        # Get top leads for display
        top_leads = results_df.nlargest(50, 'conversion_probability').to_dict('records')
        
        return render_template('results.html', leads=top_leads, stats=stats)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error details:\n{error_detail}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/download/<filename>')
def download(filename):
    """Download results file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)