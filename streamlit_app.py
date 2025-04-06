import streamlit as st
from datetime import datetime
import pandas as pd
import json

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
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Chat Interface", "Underutilized Carriers", "Email Generator"])

# Chat Interface
if app_mode == "Chat Interface":
    st.header("Carrier Information Chat")
    st.markdown("Ask questions about carrier usage, capacity, or contacts.")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for sender, message, time in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f"**You ({time}):** {message}")
            else:
                st.markdown(f"**Assistant ({time}):** {message}")
    
    # User input
    user_query = st.text_input("Ask a question about carriers:", key="user_query")
    if st.button("Submit") and user_query:
        # Add user query to chat history
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append(("user", user_query, current_time))
        
        # Get bot response
        try:
            bot_response = chatbot.answer_query(user_query)
            st.session_state.chat_history.append(("bot", bot_response, current_time))
            
            # Rerun to update the display
            st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Underutilized Carriers Report
elif app_mode == "Underutilized Carriers":
    st.header("Underutilized Carriers Report")
    st.markdown(f"Showing carriers with usage below {system_components['analyzer'].USAGE_THRESHOLD_PERCENT}% of capacity")
    
    # Get underutilized carriers
    underutilized = analyzer.identify_underutilized_carriers(df_merged)
    
    if underutilized is None or underutilized.empty:
        st.success("No underutilized carriers found!")
    else:
        # Display the table
        st.dataframe(underutilized[[
            'standardized_carrier_name', 'configured_capacity', 
            'peak_usage', 'usage_percentage', 'proposed_capacity'
        ]].sort_values('usage_percentage'))
        
        # Show statistics
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Total Carriers", len(df_merged['standardized_carrier_name'].unique()))
        col2.metric("Underutilized Carriers", len(underutilized))

# Email Generator
elif app_mode == "Email Generator":
    st.header("Capacity Reduction Email Generator")
    st.markdown("Generate emails for underutilized carriers")
    
    # Get underutilized carriers
    underutilized = analyzer.identify_underutilized_carriers(df_merged)
    
    if underutilized is None or underutilized.empty:
        st.success("No underutilized carriers to notify!")
    else:
        # Carrier selection
        selected_carrier = st.selectbox(
            "Select Carrier",
            underutilized['standardized_carrier_name'].unique()
        )
        
        if st.button("Generate Email"):
            carrier_info = underutilized[underutilized['standardized_carrier_name'] == selected_carrier].iloc[0].to_dict()
            email_result = emailer.generate_capacity_reduction_email(carrier_info)
            
            if 'error' in email_result:
                st.error(email_result['error'])
            else:
                # Store the generated email
                st.session_state.generated_emails[selected_carrier] = email_result
                
                # Display the email
                st.subheader(f"Email for {selected_carrier}")
                st.markdown(f"**To:** {email_result['to']}")
                st.markdown(f"**CC:** {', '.join(email_result['cc'])}")
                st.markdown(f"**Subject:** {email_result['subject']}")
                st.markdown("**Body:**")
                st.text(email_result['body'])
                
                # Download button
                email_json = json.dumps(email_result, indent=2)
                st.download_button(
                    label="Download Email as JSON",
                    data=email_json,
                    file_name=f"capacity_reduction_{selected_carrier}.json",
                    mime="application/json"
                )
