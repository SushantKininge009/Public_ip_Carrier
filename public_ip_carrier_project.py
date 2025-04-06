# Paste all your notebook code here (everything except the Streamlit part)
# Enhanced imports
import pandas as pd
import io
import re
from fuzzywuzzy import process
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from datetime import datetime
import json
from typing import Dict, List, Optional

# Configuration (move to a config file or environment variables in production)
class Config:
    PROJECT_ID = "gen-ai-rajan-labs"
    LOCATION = "us-central1"
    GCS_BUCKET_NAME = "publicip_carrier_data"
    PEAK_USAGE_FILE = "carrier_peak_usage.xlsx"
    ACCOUNT_MANAGERS_FILE = "carrier_account_managers.xlsx"
    SUPPORT_FILE = "carrier_first_line_support.xlsx"
    USAGE_THRESHOLD_PERCENT = 40.0
    CAPACITY_REDUCTION_FACTOR = 0.5
    MODEL_NAME = "gemini-1.0-pro"
    
# Initialize clients
vertexai.init(project=Config.PROJECT_ID, location=Config.LOCATION)
storage_client = storage.Client(project=Config.PROJECT_ID)
bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
model = GenerativeModel(Config.MODEL_NAME)

class DataProcessor:
    """Handles data loading, cleaning, and standardization"""
    
    @staticmethod
    def load_excel_from_gcs(blob_name: str) -> Optional[pd.DataFrame]:
        """Downloads an Excel file from GCS and loads it into a pandas DataFrame."""
        try:
            blob = bucket.blob(blob_name)
            content = blob.download_as_bytes()
            df = pd.read_excel(io.BytesIO(content))
            print(f"✅ Successfully loaded: {blob_name}")
            return df
        except Exception as e:
            print(f"❌ Error loading {blob_name}: {str(e)}")
            return None
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names (lowercase, replace spaces with underscores)."""
        if df is None:
            return None
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)
        return df
    
    @staticmethod
    def standardize_carrier_names(df: pd.DataFrame, reference_names: List[str], 
                                carrier_col: str = 'carrier_name', threshold: int = 85) -> pd.DataFrame:
        """Standardizes carrier names using fuzzy matching against reference names."""
        if df is None or carrier_col not in df.columns:
            return df
            
        standardized_names = {}
        for name in df[carrier_col].unique():
            if pd.isna(name):
                standardized_names[name] = None
                continue
                
            match, score = process.extractOne(str(name), reference_names)
            standardized_names[name] = match if score >= threshold else str(name)
        
        df['standardized_carrier_name'] = df[carrier_col].map(standardized_names)
        return df

class CarrierAnalyzer:
    """Performs analysis on carrier data"""
    
    @staticmethod
    def identify_underutilized_carriers(df_merged: pd.DataFrame) -> pd.DataFrame:
        """Identifies carriers with usage below threshold and calculates proposed capacity"""
        if df_merged is None:
            return None
            
        df = df_merged.copy()
        
        # Ensure numeric types
        numeric_cols = ['peak_usage', 'configured_capacity']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing/invalid data
        df = df.dropna(subset=numeric_cols)
        df = df[df['configured_capacity'] > 0]
        
        # Calculate usage percentage
        df['usage_percentage'] = (df['peak_usage'] / df['configured_capacity']) * 100
        
        # Identify underutilized carriers
        underutilized = df[df['usage_percentage'] < Config.USAGE_THRESHOLD_PERCENT].copy()
        
        # Calculate proposed capacity
        underutilized['proposed_capacity'] = (underutilized['configured_capacity'] * 
                                            Config.CAPACITY_REDUCTION_FACTOR).round().astype(int)
        
        return underutilized

class EmailGenerator:
    """Handles email generation and management"""
    
    @staticmethod
    def generate_capacity_reduction_email(carrier_info: Dict) -> Dict:
        """Generates email content for capacity reduction notification"""
        required_fields = [
            'standardized_carrier_name', 'configured_capacity', 'peak_usage',
            'usage_percentage', 'proposed_capacity', 'carrier_company_account_manager_email'
        ]
        
        # Validate required fields
        for field in required_fields:
            if field not in carrier_info or pd.isna(carrier_info[field]):
                return {"error": f"Missing required field: {field}"}
                
        # Prepare context for LLM
        context = {
            "carrier_name": carrier_info.get('standardized_carrier_name', 'N/A'),
            "current_capacity": carrier_info.get('configured_capacity', 'N/A'),
            "peak_usage": carrier_info.get('peak_usage', 'N/A'),
            "usage_percent": carrier_info.get('usage_percentage', 'N/A'),
            "new_capacity": carrier_info.get('proposed_capacity', 'N/A'),
            "your_am_name": carrier_info.get('your_company_account_manager_name', 'Your Account Manager'),
            "your_am_email": carrier_info.get('your_company_account_manager_email', 'your_am@yourcompany.com'),
            "carrier_am_name": carrier_info.get('carrier_company_account_manager_name', 'Carrier Contact'),
            "carrier_am_email": carrier_info.get('carrier_company_account_manager_email'),
            "support_name": carrier_info.get('first_line_contact_name', 'Carrier Support'),
            "support_email": carrier_info.get('first_line_contact_email', ''),
            "threshold": Config.USAGE_THRESHOLD_PERCENT,
            "reduction_factor": int(Config.CAPACITY_REDUCTION_FACTOR * 100)
        }
        
        prompt = f"""
        Generate a professional email notification regarding a planned capacity reduction for a Public IP Carrier voice trunk.

        **Instructions:**
        1. Be polite and professional.
        2. Clearly state the reason for the reduction (peak usage consistently below {context['threshold']}% of configured capacity).
        3. Include specific numbers: current capacity {context['current_capacity']}, peak usage {context['peak_usage']:.0f} ({context['usage_percent']:.1f}%), proposed new capacity {context['new_capacity']}.
        4. Address the email to {context['carrier_am_name']} at {context['carrier_am_email']}.
        5. CC {context['your_am_name']} ({context['your_am_email']}) and {context['support_name']} ({context['support_email']}).
        6. Provide contact information for questions.
        7. Suggest a timeframe for discussion (e.g., "within the next two weeks").
        8. Use a professional tone suitable for telecom industry communication.

        **Output Format:**
        Subject: [Clear subject line]
        Body: [Email body content]
        """
        
        try:
            response = model.generate_content(prompt)
            email_text = response.text
            
            # Parse response
            subject = f"Planned Capacity Adjustment for {context['carrier_name']} Voice Trunk"
            body = email_text
            
            # Extract subject if model provides it
            if "Subject:" in email_text:
                subject = email_text.split("Subject:")[1].split("\n")[0].strip()
                body = email_text.split("Body:")[1] if "Body:" in email_text else email_text
            
            return {
                "to": context['carrier_am_email'],
                "cc": [e for e in [context['your_am_email'], context['support_email']] if pd.notna(e) and '@' in str(e)],
                "subject": subject,
                "body": body.strip(),
                "carrier": context['carrier_name']
            }
        except Exception as e:
            return {"error": f"Error generating email: {str(e)}"}

class Chatbot:
    """Handles user queries about carriers"""
    
    def __init__(self, df_merged: pd.DataFrame):
        self.df = df_merged
        
    def answer_query(self, query: str) -> str:
        """Processes user query and generates response"""
        if self.df is None:
            return "Data not available. Please load data first."
            
        # Check for specific carrier queries
        carrier_match = re.search(r'(carrier|provider|trunk)\\s+([A-Za-z0-9_\\-\\s]+)', query, re.IGNORECASE)
        if carrier_match:
            carrier_name = carrier_match.group(2).strip()
            return self._answer_carrier_specific_query(carrier_name, query)
            
        # Check for general statistics
        if any(word in query.lower() for word in ['how many', 'count', 'number', 'statistics']):
            return self._answer_statistical_query(query)
            
        # Default to general LLM response
        return self._generate_llm_response(query)
    
    def _answer_carrier_specific_query(self, carrier_name: str, query: str) -> str:
        """Handles queries about specific carriers"""
        # Find matching carrier (fuzzy match if needed)
        matches = process.extract(carrier_name, self.df['standardized_carrier_name'].unique(), limit=3)
        
        if not matches or matches[0][1] < 70:  # Low confidence in match
            return f"I couldn't find a carrier matching '{carrier_name}'. Please check the name and try again."
            
        best_match = matches[0][0]
        carrier_data = self.df[self.df['standardized_carrier_name'] == best_match]
        
        if carrier_data.empty:
            return f"Found carrier {best_match} but no data available."
            
        # Prepare context for LLM
        context = {
            "carrier": best_match,
            "data": carrier_data.iloc[0].to_dict(),
            "query": query
        }
        
        prompt = f"""
        You are a telecom solutions assistant. Answer the user's question about a specific carrier using the provided data.
        
        Carrier: {context['carrier']}
        User Query: {context['query']}
        
        Available Data:
        {json.dumps(context['data'], indent=2)}
        
        Provide a concise, professional answer focusing on the requested information.
        If the query is about capacity reduction eligibility, mention the {Config.USAGE_THRESHOLD_PERCENT}% threshold rule.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error processing your query: {str(e)}"
    
    def _answer_statistical_query(self, query: str) -> str:
        """Answers statistical questions about the carriers"""
        total_carriers = len(self.df['standardized_carrier_name'].unique())
        underutilized = CarrierAnalyzer.identify_underutilized_carriers(self.df)
        underutilized_count = len(underutilized) if underutilized is not None else 0
        
        stats = {
            "total_carriers": total_carriers,
            "underutilized_count": underutilized_count,
            "threshold": Config.USAGE_THRESHOLD_PERCENT
        }
        
        prompt = f"""
        Answer the user's statistical question using this data:
        
        Statistics:
        {json.dumps(stats, indent=2)}
        
        User Query: {query}
        
        Provide a clear, numerical response first, then any additional context.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating statistics: {str(e)}"
    
    def _generate_llm_response(self, query: str) -> str:
        """Generates a general response using LLM"""
        prompt = f"""
        You are a telecom solutions architect assistant. Answer the user's question professionally.
        
        Available Data Overview:
        - {len(self.df['standardized_carrier_name'].unique())} carriers
        - Data includes: peak usage, configured capacity, account managers, support contacts
        - Capacity reduction rule: below {Config.USAGE_THRESHOLD_PERCENT}% usage → reduce to 50% capacity
        
        User Query: {query}
        
        If the question requires specific carrier data that isn't provided, say so.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error processing your query: {str(e)}"

# Main workflow
def main_workflow():
    """Loads data, processes it, and makes components available"""
    print("Loading and processing data...")
    
    # Load data
    df_usage = DataProcessor.load_excel_from_gcs(Config.PEAK_USAGE_FILE)
    df_managers = DataProcessor.load_excel_from_gcs(Config.ACCOUNT_MANAGERS_FILE)
    df_support = DataProcessor.load_excel_from_gcs(Config.SUPPORT_FILE)
    
    # Clean column names
    df_usage = DataProcessor.clean_column_names(df_usage)
    df_managers = DataProcessor.clean_column_names(df_managers)
    df_support = DataProcessor.clean_column_names(df_support)
    
    # Standardize names (using managers as reference)
    reference_names = df_managers['carrier_name'].dropna().unique().tolist()
    df_usage = DataProcessor.standardize_carrier_names(df_usage, reference_names, 'carrier_name')
    df_support = DataProcessor.standardize_carrier_names(df_support, reference_names, 'carrier_name')
    df_managers['standardized_carrier_name'] = df_managers['carrier_name']
    
    # Merge data
    df_merged = df_usage.merge(
        df_managers[['standardized_carrier_name', 'your_company_account_manager_name', 
                    'your_company_account_manager_email', 'carrier_company_account_manager_name',
                    'carrier_company_account_manager_email']].drop_duplicates(),
        on='standardized_carrier_name',
        how='left'
    ).merge(
        df_support[['standardized_carrier_name', 'first_line_contact_name', 
                   'first_line_contact_email']].drop_duplicates(),
        on='standardized_carrier_name',
        how='left'
    )
    
    # Initialize components
    analyzer = CarrierAnalyzer()
    emailer = EmailGenerator()
    chatbot = Chatbot(df_merged)
    
    return {
        "merged_data": df_merged,
        "analyzer": analyzer,
        "emailer": emailer,
        "chatbot": chatbot
    }

# Initialize the system
system_components = main_workflow()
