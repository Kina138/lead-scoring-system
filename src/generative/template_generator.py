"""
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
