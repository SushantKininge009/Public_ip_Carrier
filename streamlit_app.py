
# streamlit_app.py - Updated version
import streamlit as st
from datetime import datetime
import pandas as pd
import json
import os

# App configuration
st.set_page_config(
    page_title="Public IP Carrier Manager",
    page_icon="📞",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated_emails' not in st.session_state:
    st.session_state.generated_emails = {}

# App header
st.title("Public IP Carrier Management Portal")
st.markdown("""
    **Telecom Solutions Architect Tool** for managing carrier capacity, contacts, and communications.
""")

# Get system components from the imported notebook code
try:
    from public_ip_carrier_project import system_components
    df_merged = system_components["merged_data"]
    chatbot = system_components["chatbot"]
    emailer = system_components["emailer"]
    analyzer = system_components["analyzer"]
    
    # Display success message
    st.success("System components loaded successfully!")
    
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.warning("Running in demo mode with sample data")
    
    # Create sample data for demo purposes
    sample_data = {
        'standardized_carrier_name': ['Carrier A', 'Carrier B', 'Carrier C'],
        'configured_capacity': [1000, 2000, 1500],
        'peak_usage': [300, 500, 700],
        'usage_percentage': [30.0, 25.0, 46.7],
        'carrier_company_account_manager_name': ['John Doe', 'Jane Smith', 'Mike Johnson'],
        'carrier_company_account_manager_email': ['john@carriera.com', 'jane@carrierb.com', 'mike@carrierc.com'],
        'first_line_contact_name': ['Support A', 'Support B', 'Support C'],
        'first_line_contact_email': ['support@carriera.com', 'support@carrierb.com', 'support@carrierc.com']
    }
    
    df_merged = pd.DataFrame(sample_data)
    
    # Create mock components
    class MockAnalyzer:
        def identify_underutilized_carriers(self, df):
            df['proposed_capacity'] = (df['configured_capacity'] * 0.5).astype(int)
            return df[df['usage_percentage'] < 40]
    
    class MockEmailer:
        def generate_capacity_reduction_email(self, carrier_info):
            return {
                "to": carrier_info['carrier_company_account_manager_email'],
                "cc": [carrier_info['first_line_contact_email']],
                "subject": f"Capacity Adjustment for {carrier_info['standardized_carrier_name']}",
                "body": f"Dear {carrier_info['carrier_company_account_manager_name']},\n\nWe propose reducing capacity...",
                "carrier": carrier_info['standardized_carrier_name']
            }
    
    class MockChatbot:
        def answer_query(self, query):
            return f"Sample response to: {query}"
    
    analyzer = MockAnalyzer()
    emailer = MockEmailer()
    chatbot = MockChatbot()
