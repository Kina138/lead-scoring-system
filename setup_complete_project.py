"""
Complete Project Setup Script
Tạo tất cả các file code còn lại cần thiết
"""

import os
from pathlib import Path

def create_file(filepath, content):
    """Create file with content"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Created: {filepath}")

# =============================================================================
# SHAP EXPLAINER
# =============================================================================
shap_explainer_code = '''"""
SHAP-based explainability module
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import print_section, ensure_dir


class SHAPExplainer:
    """SHAP explainability for lead scoring models"""
    
    def __init__(self, model, X_background, model_type='tree'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP
            model_type: 'tree' or 'kernel'
        """
        self.model = model
        self.X_background = X_background
        self.model_type = model_type
        
        # Initialize appropriate explainer
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_background, 100)
            )
        
        print(f"✓ SHAP {model_type} explainer initialized")
    
    def explain_predictions(self, X):
        """Calculate SHAP values for predictions"""
        shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        return shap_values
    
    def plot_summary(self, X, feature_names, max_display=20):
        """Create SHAP summary plot"""
        print_section("Generating SHAP Summary Plot")
        
        shap_values = self.explain_predictions(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        ensure_dir('outputs/visualizations')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ SHAP summary plot saved")
    
    def plot_waterfall(self, X_instance, feature_names, instance_idx=0):
        """Create SHAP waterfall plot for single prediction"""
        shap_values = self.explain_predictions(X_instance)
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=X_instance[instance_idx],
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        
        plt.tight_layout()
        plt.savefig(f'outputs/visualizations/shap_waterfall_{instance_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Waterfall plot saved for instance {instance_idx}")
    
    def get_feature_importance(self, X, feature_names):
        """Get feature importance DataFrame"""
        shap_values = self.explain_predictions(X)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot top N important features"""
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Features by SHAP Importance', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/shap_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved")
'''

# =============================================================================
# TEMPLATE GENERATOR
# =============================================================================
template_generator_code = '''"""
Template-based marketing recommendation generator
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import print_section


class TemplateGenerator:
    """Generate marketing recommendations using templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize recommendation templates"""
        return {
            'High': {
                'subject': '🎯 Special Offer: Your Learning Journey Starts Now',
                'message_template': """Dear {name},

Based on your exceptional interest in our {specialization} program and {engagement} engagement, 
we're excited to offer you a personalized enrollment path.

✨ Key Benefits for You:
• {benefit_1}
• {benefit_2}
• {benefit_3}

🎯 Next Step: Schedule a 1-on-1 consultation this week with our enrollment advisor.

We're committed to helping you achieve your career goals.

Best Regards,
X Education Team""",
                'channel': 'Email + Phone Call',
                'timing': 'Within 24 hours',
                'priority': 'URGENT - Immediate follow-up required',
                'action': 'Schedule personal consultation call'
            },
            'Medium': {
                'subject': '📚 Continue Your Learning Journey',
                'message_template': """Hi {name},

Thank you for your interest in our {specialization} program. 

📖 We recommend:
• Reviewing our curriculum and success stories
• Attending our upcoming webinar on {topic}
• Connecting with our advisors for questions

🎯 Next Step: Join our free webinar to learn about career opportunities in {specialization}.

Looking forward to supporting your goals.

Best Regards,
X Education Team""",
                'channel': 'Email',
                'timing': 'Within 2-3 days',
                'priority': 'STANDARD - Regular follow-up',
                'action': 'Send webinar invitation + course materials'
            },
            'Low': {
                'subject': '💡 Exploring Your Options? Resources Inside',
                'message_template': """Hello {name},

We understand that choosing the right program is an important decision.

📚 Free Resources:
• Guide: "Mastering the Interview"
• {specialization} alumni success stories
• Flexible payment options info

🎯 Next Step: Download our free resources and stay updated on industry trends.

Take your time exploring - we're here when you're ready.

Best Regards,
X Education Team""",
                'channel': 'Email (nurture sequence)',
                'timing': 'Within 7 days',
                'priority': 'LOW - Nurture campaign',
                'action': 'Add to email nurture sequence'
            }
        }
    
    def generate_recommendation(self, lead_data, segment):
        """
        Generate recommendation for a single lead
        
        Args:
            lead_data: Dict with lead information
            segment: 'High', 'Medium', or 'Low'
        
        Returns:
            dict: Recommendation details
        """
        template = self.templates[segment]
        
        # Extract lead info
        name = lead_data.get('name', 'Valued Lead')
        specialization = lead_data.get('Specialization', 'our programs')
        
        # Determine engagement level
        time_spent = lead_data.get('Total Time Spent on Website', 0)
        if time_spent > 900:
            engagement = 'exceptional'
        elif time_spent > 500:
            engagement = 'high'
        else:
            engagement = 'moderate'
        
        # Get benefits
        benefits = self._get_benefits(specialization)
        
        # Format message
        message = template['message_template'].format(
            name=name,
            specialization=specialization,
            engagement=engagement,
            benefit_1=benefits[0],
            benefit_2=benefits[1],
            benefit_3=benefits[2],
            topic='Career Development' if specialization == 'our programs' else specialization
        )
        
        return {
            'segment': segment,
            'email_subject': template['subject'],
            'message': message,
            'channel': template['channel'],
            'timing': template['timing'],
            'priority': template['priority'],
            'recommended_action': template['action'],
            'conversion_probability': lead_data.get('conversion_probability', 0)
        }
    
    def _get_benefits(self, specialization):
        """Get benefits based on specialization"""
        benefits_map = {
            'IT': [
                'Industry-recognized certifications',
                'Hands-on projects with cutting-edge tools',
                'Career support and placement assistance'
            ],
            'Finance': [
                'CFA-aligned curriculum',
                'Financial modeling and analytics',
                'Industry expert mentorship'
            ],
            'Marketing': [
                'Digital marketing certification',
                'Real-world campaign projects',
                'Marketing analytics tools training'
            ],
            'HR': [
                'SHRM-aligned content',
                'Talent management frameworks',
                'Organizational behavior insights'
            ]
        }
        
        default_benefits = [
            'Expert-led instruction',
            'Flexible learning schedule',
            'Lifetime access to materials'
        ]
        
        return benefits_map.get(specialization, default_benefits)
    
    def generate_batch_recommendations(self, leads_df):
        """Generate recommendations for multiple leads"""
        print_section("Generating Marketing Recommendations")
        
        recommendations = []
        
        for idx, row in leads_df.iterrows():
            lead_dict = row.to_dict()
            segment = row.get('segment', 'Medium')
            
            rec = self.generate_recommendation(lead_dict, segment)
            rec['lead_index'] = idx
            recommendations.append(rec)
        
        rec_df = pd.DataFrame(recommendations)
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        print(f"  High priority: {len(rec_df[rec_df['segment'] == 'High'])}")
        print(f"  Medium priority: {len(rec_df[rec_df['segment'] == 'Medium'])}")
        print(f"  Low priority: {len(rec_df[rec_df['segment'] == 'Low'])}")
        
        return rec_df
'''

# Create files
print("Creating remaining project files...")
print("="*60)

create_file('src/explainability/shap_explainer.py', shap_explainer_code)
create_file('src/generative/template_generator.py', template_generator_code)

print("="*60)
print("✓ All core files created successfully!")

