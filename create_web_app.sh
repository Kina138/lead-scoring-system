#!/bin/bash

# Create Flask App
cat > web/app.py << 'FLASK_EOF'
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
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load best model and preprocessor
try:
    model = joblib.load('../models/gradient_boosting.pkl')
    preprocessor = DataPreprocessor()
    preprocessor.load('../models/preprocessor.pkl')
    generator = TemplateGenerator()
    print("✓ Models loaded successfully")
except:
    model, preprocessor, generator = None, None, None
    print("⚠ Models not found. Please train models first.")

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
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        predictions = model.predict_proba(X)[:, 1]
        
        # Add results to dataframe
        df['conversion_probability'] = predictions
        df['segment'] = df['conversion_probability'].apply(classify_segment)
        
        # Generate recommendations
        recommendations_df = generator.generate_batch_recommendations(df)
        
        # Combine results
        results_df = pd.concat([df, recommendations_df], axis=1)
        
        # Save results
        output_path = os.path.join('outputs/predictions', f'results_{filename}')
        results_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        stats = {
            'total_leads': len(df),
            'high_priority': len(df[df['segment'] == 'High']),
            'medium_priority': len(df[df['segment'] == 'Medium']),
            'low_priority': len(df[df['segment'] == 'Low']),
            'avg_score': predictions.mean(),
            'filename': f'results_{filename}'
        }
        
        # Get top leads for display
        top_leads = results_df.nlargest(50, 'conversion_probability').to_dict('records')
        
        return render_template('results.html', leads=top_leads, stats=stats)
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/download/<filename>')
def download(filename):
    """Download results file"""
    filepath = os.path.join('outputs/predictions', filename)
    return send_file(filepath, as_attachment=True)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
FLASK_EOF

# Create HTML Templates
cat > web/templates/base.html << 'BASE_EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}AI Lead Scoring System{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">🎯 AI Lead Scoring</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('index') }}">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('upload') }}">Upload Data</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('about') }}">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </div>

    <footer class="mt-5 py-3 bg-light">
        <div class="container text-center">
            <p class="text-muted mb-0">© 2025 AI Lead Scoring System | CS 687 Capstone Project</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
BASE_EOF

cat > web/templates/index.html << 'INDEX_EOF'
{% extends "base.html" %}

{% block content %}
<div class="row mt-5">
    <div class="col-md-8 offset-md-2">
        <div class="card shadow-lg">
            <div class="card-body text-center p-5">
                <h1 class="display-4 mb-4">Welcome to AI Lead Scoring</h1>
                <p class="lead mb-4">Intelligent lead prioritization powered by machine learning</p>
                
                <div class="row mt-5 mb-4">
                    <div class="col-md-4">
                        <div class="feature-box p-4">
                            <div class="feature-icon mb-3">🔮</div>
                            <h5>Predict Conversions</h5>
                            <p class="text-muted">AI models predict lead quality with 90%+ accuracy</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="feature-box p-4">
                            <div class="feature-icon mb-3">📊</div>
                            <h5>Explain Results</h5>
                            <p class="text-muted">SHAP analysis shows why leads are prioritized</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="feature-box p-4">
                            <div class="feature-icon mb-3">✉️</div>
                            <h5>Generate Actions</h5>
                            <p class="text-muted">Auto-create personalized marketing messages</p>
                        </div>
                    </div>
                </div>
                
                <a href="{{ url_for('upload') }}" class="btn btn-primary btn-lg mt-3">
                    Get Started →
                </a>
            </div>
        </div>
    </div>
</div>
{% endblock %}
INDEX_EOF

cat > web/templates/upload.html << 'UPLOAD_EOF'
{% extends "base.html" %}

{% block content %}
<div class="row mt-5">
    <div class="col-md-6 offset-md-3">
        <div class="card shadow">
            <div class="card-header bg-primary text-white">
                <h4 class="mb-0">📤 Upload Lead Data</h4>
            </div>
            <div class="card-body p-4">
                <form method="POST" enctype="multipart/form-data">
                    <div class="mb-4">
                        <label class="form-label">Select CSV File</label>
                        <input type="file" class="form-control" name="file" accept=".csv" required>
                        <div class="form-text">Upload your lead data in CSV format (max 16MB)</div>
                    </div>
                    
                    <div class="alert alert-info">
                        <strong>Required columns:</strong>
                        <ul class="mb-0">
                            <li>Lead Origin, Lead Source</li>
                            <li>TotalVisits, Total Time Spent on Website</li>
                            <li>Specialization, Last Activity</li>
                        </ul>
                    </div>
                    
                    <button type="submit" class="btn btn-primary w-100 mt-3">
                        Upload and Analyze
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
UPLOAD_EOF

cat > web/templates/results.html << 'RESULTS_EOF'
{% extends "base.html" %}

{% block content %}
<div class="row mt-4">
    <div class="col-12">
        <h2>📊 Lead Scoring Results</h2>
    </div>
</div>

<div class="row mt-3">
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <h6 class="card-title text-muted">Total Leads</h6>
                <h2 class="text-primary">{{ stats.total_leads }}</h2>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <h6 class="card-title text-muted">High Priority</h6>
                <h2 class="text-success">{{ stats.high_priority }}</h2>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <h6 class="card-title text-muted">Medium Priority</h6>
                <h2 class="text-warning">{{ stats.medium_priority }}</h2>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card text-center">
            <div class="card-body">
                <h6 class="card-title text-muted">Avg Score</h6>
                <h2 class="text-info">{{ "%.0f"|format(stats.avg_score * 100) }}%</h2>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">Top 50 Leads</h5>
                <a href="{{ url_for('download', filename=stats.filename) }}" class="btn btn-success btn-sm">
                    ⬇ Download Full Results
                </a>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Score</th>
                                <th>Segment</th>
                                <th>Specialization</th>
                                <th>Channel</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for lead in leads %}
                            <tr>
                                <td>{{ loop.index }}</td>
                                <td>
                                    <span class="badge 
                                        {% if lead.conversion_probability >= 0.7 %}bg-success
                                        {% elif lead.conversion_probability >= 0.3 %}bg-warning
                                        {% else %}bg-secondary{% endif %}">
                                        {{ "%.0f"|format(lead.conversion_probability * 100) }}%
                                    </span>
                                </td>
                                <td>{{ lead.segment }}</td>
                                <td>{{ lead.Specialization if lead.Specialization else 'N/A' }}</td>
                                <td><small>{{ lead.channel }}</small></td>
                                <td><small>{{ lead.recommended_action }}</small></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
RESULTS_EOF

cat > web/templates/about.html << 'ABOUT_EOF'
{% extends "base.html" %}

{% block content %}
<div class="row mt-4">
    <div class="col-md-8 offset-md-2">
        <h2>About This System</h2>
        
        <div class="card mt-4">
            <div class="card-body">
                <h4>🎯 Project Overview</h4>
                <p>AI-Based Lead Scoring and Generative Marketing Recommendation System</p>
                <p>This system combines predictive analytics, explainable AI, and generative recommendations to help organizations prioritize leads and optimize marketing strategies.</p>
                
                <h4 class="mt-4">🔬 Technical Stack</h4>
                <ul>
                    <li><strong>Machine Learning:</strong> Logistic Regression, Random Forest, Gradient Boosting, Neural Network</li>
                    <li><strong>Explainability:</strong> SHAP (SHapley Additive exPlanations)</li>
                    <li><strong>Web Framework:</strong> Flask</li>
                    <li><strong>Data Processing:</strong> Pandas, NumPy, Scikit-learn</li>
                </ul>
                
                <h4 class="mt-4">👨‍🎓 Author</h4>
                <p><strong>Anh Thi Van Bui</strong><br>
                MS in Computer Science<br>
                City University of Seattle<br>
                Advisor: Sivakumar Visweswaran</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}
ABOUT_EOF

# Create CSS
cat > web/static/css/style.css << 'CSS_EOF'
.feature-box {
    transition: transform 0.3s;
    border-radius: 10px;
    background: #f8f9fa;
}

.feature-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.feature-icon {
    font-size: 3rem;
}

.card {
    border-radius: 10px;
}

.badge {
    font-size: 0.9rem;
    padding: 0.4rem 0.6rem;
}

footer {
    margin-top: auto;
}

body {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}
CSS_EOF

echo "✓ Flask web application created successfully!"

