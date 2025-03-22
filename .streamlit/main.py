# This file helps Streamlit Cloud find the main application file
# It imports and runs the main streamlit_cloud_app.py file

import sys
import os

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main app
from streamlit_cloud_app import main

# Run the main function
if __name__ == "__main__":
    main() 