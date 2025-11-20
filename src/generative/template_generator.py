"""
Template generator for personalized lead recommendations
"""
import pandas as pd
from typing import Dict, List

class TemplateGenerator:
    """Generate personalized recommendations for leads"""
    
    def __init__(self):
        """Initialize template generator"""
        self.templates = {
            'High': {
                'email': "Immediate follow-up required. Priority lead with high conversion potential.",
                'action': "Schedule call within 24 hours. Assign to senior sales rep.",
                'channel': "Phone call + Personalized email"
            },
            'Medium': {
                'email': "Promising lead. Follow-up recommended within 3 days.",
                'action': "Send personalized email with relevant course information.",
                'channel': "Email + Follow-up call"
            },
            'Low': {
                'email': "Add to nurture campaign. Monitor engagement over time.",
                'action': "Add to automated email sequence. Re-evaluate in 30 days.",
                'channel': "Automated email campaign"
            }
        }
    
    def generate_recommendation(self, lead_data: Dict) -> str:
        """
        Generate personalized recommendation for a single lead
        
        Args:
            lead_data: Dictionary containing lead information
            
        Returns:
            Recommendation string
        """
        segment = lead_data.get('segment', 'Low')
        score = lead_data.get('conversion_probability', 0)
        
        # Get base template
        template = self.templates.get(segment, self.templates['Low'])
        
        # Personalize based on lead data
        specialization = lead_data.get('Specialization', 'Not specified')
        lead_source = lead_data.get('Lead Source', 'Unknown')
        total_time = lead_data.get('Total Time Spent on Website', 0)
        
        # Build recommendation
        recommendation = f"Score: {score*100:.1f}% | Segment: {segment}\n"
        recommendation += f"Action: {template['action']}\n"
        recommendation += f"Channel: {template['channel']}\n"
        
        # Add personalization
        if specialization and str(specialization) != 'nan' and str(specialization) != 'Select':
            recommendation += f"Focus: Highlight {specialization} program benefits\n"
        
        if lead_source and str(lead_source) != 'nan':
            recommendation += f"Context: Lead from {lead_source}\n"
        
        if total_time and total_time > 100:
            recommendation += f"Note: High engagement detected ({int(total_time)} mins on site)\n"
        
        return recommendation
    
    def generate_batch_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate recommendations for batch of leads
        
        Args:
            df: DataFrame with lead data (must include 'segment' and 'conversion_probability')
            
        Returns:
            DataFrame with recommendations
        """
        recommendations = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict
                lead_dict = row.to_dict()
                
                # Generate recommendation
                rec = self.generate_recommendation(lead_dict)
                recommendations.append(rec)
            except Exception as e:
                # Fallback recommendation
                segment = row.get('segment', 'Low')
                recommendations.append(f"Segment: {segment} | Standard follow-up recommended.")
        
        # Return as DataFrame
        result_df = pd.DataFrame({
            'recommendation': recommendations
        })
        
        return result_df