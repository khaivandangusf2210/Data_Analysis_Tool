"""
Main entry point for Streamlit Cloud deployment.
This file serves as the default entry point that Streamlit Cloud looks for.
"""
import streamlit as st

try:
    # Import the main function from the streamlit_cloud_app module
    from streamlit_cloud_app import main
    
    # Run the main function when this script is executed
    if __name__ == "__main__":
        main()
except ImportError as e:
    st.error(f"Error importing streamlit_cloud_app: {e}")
    st.info("Please ensure streamlit_cloud_app.py exists in the project root.") 