# Data Analysis Tool

A powerful web application for data analysis and visualization built with Streamlit.

## Live Demo

The app is deployed and accessible at: **[https://data-analysis-io.streamlit.app/](https://data-analysis-io.streamlit.app/)**

## Features

- **Data Upload**: Support for CSV and Excel files with optimized loading for large datasets
- **Data Analysis**:
  - Statistical Summary
  - Missing Values Analysis
  - Correlation Analysis with heatmaps
  - Distribution Analysis with histograms and KDE
  - Outlier Detection
- **Visualizations**:
  - Interactive Scatter Plots
  - 3D Scatter Plots
  - Distribution Plots
  - Box Plots
  - Time Series Analysis
  - Interactive controls and tooltips

## Usage

1. Upload your data file (CSV or Excel)
2. Select columns for analysis
3. Navigate through the Data Overview, Visualization, and Statistical Analysis tabs
4. Explore visualizations with interactive controls

## Requirements

The application requires the following main dependencies:
- streamlit==1.29.0
- pandas==2.0.3
- numpy>=1.26.0
- plotly==5.15.0
- scipy==1.11.3
- statsmodels==0.14.0

See `requirements.txt` for complete dependencies.

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run streamlit_app.py`

## Project Structure

- `streamlit_app.py`: Entry point file for Streamlit Cloud deployment
- `streamlit_cloud_app.py`: Main application file with all functionality
- `streamlit_config.yaml`: Configuration for Streamlit Cloud deployment
- `assets/`: Directory containing application assets
- `requirements.txt`: List of Python dependencies
- `.streamlit/`: Configuration directory for Streamlit settings
- `.gitignore`: Configuration to exclude unnecessary files from version control

## Features in Detail

### Data Loading
- Support for CSV and Excel files
- Automatic data type conversion and optimization

### Data Overview
- Data overview with shape and preview
- Data type information
- Missing value analysis
- Unique value counts for categorical variables

### Visualization
- **Correlation Analysis**: Interactive heatmaps with top correlations display
- **Distribution Analysis**: Histograms with KDE overlay
- **Scatter Plots**: 2D and 3D with color and size dimensions
- **Box Plots**: For outlier detection with optional grouping
- **Time Series Analysis**: For temporal data with interactive range sliders

### Statistical Analysis
- Comprehensive descriptive statistics
- Correlation analysis with top correlations

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check if the port is already in use and kill the process if needed
3. For deployment issues, verify the following:
   - Make sure `statsmodels` is in your requirements.txt file
   - Verify your Streamlit Cloud deployment is pointing to the main branch
   - Check that the entry point file is correctly set to `streamlit_app.py`

## Deployment Notes

The application is optimized for Streamlit Cloud deployment. The main application code is in `streamlit_cloud_app.py`, but the entry point for Streamlit Cloud is `streamlit_app.py` which imports the main functionality.

The app is deployed using the following configuration:
- Main file: `streamlit_app.py`
- Python version: 3.12
- Requirements: Listed in `requirements.txt`

## Copyright

© 2025 Khai Dang. All Rights Reserved. 