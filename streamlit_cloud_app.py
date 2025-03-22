import streamlit as st

# configure the app
st.set_page_config(
    page_title="Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import os
import time
import sys
import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def process_dataframe(df):
    """Process dataframe to make it Arrow-compatible"""
    
    df_processed = df.copy()
    
    for col in df_processed.select_dtypes(include=['float32']).columns:
        df_processed[col] = df_processed[col].astype('float64')
    
   
    for col in df_processed.select_dtypes(include=['datetime64']).columns:
        df_processed[col] = df_processed[col].astype(str)
    
    
    for col in df_processed.select_dtypes(include=['object']).columns:
        try:  
            pd.to_datetime(df_processed[col], errors='raise')
            df_processed[col] = df_processed[col].astype(str)
        except:
            pass
            
    return df_processed

def analyze_data_quality(df):
    st.header("Data Types and Quality")
    
    dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    dtype_df = dtype_df.reset_index()
    dtype_df.columns = ['Column', 'Data Type']
    
    missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing_df = missing_df.reset_index()
    missing_df.columns = ['Column', 'Missing Values']
    
    quality_df = dtype_df.merge(missing_df, on='Column')
    quality_df['Missing Percentage'] = (quality_df['Missing Values'] / len(df)) * 100
    quality_df['Missing Percentage'] = quality_df['Missing Percentage'].round(2).astype(str) + '%'
    
    st.dataframe(quality_df)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.subheader("Categorical Columns")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            if unique_vals < 10:  
                st.write(f"**{col}** - {unique_vals} unique values")
                st.write(df[col].value_counts())
            else:
                st.write(f"**{col}** - {unique_vals} unique values (too many to display)")

def plot_scatter(df, numeric_cols):
    st.subheader("Scatter Plot")
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for scatter plot.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
    
    y_options = numeric_cols.copy()
    default_y_index = 0 if y_options[0] != x_col else min(1, len(y_options)-1)
    
    with col2:
        y_col = st.selectbox("Select Y-axis", y_options, 
                         index=default_y_index, key="scatter_y")
    
    col1, col2 = st.columns(2)
    
    with col1:
        color_option = st.selectbox("Select Color Variable (optional)", 
                                ["None"] + numeric_cols, key="scatter_color")
    
    with col2:
        size_option = st.selectbox("Select Size Variable (optional)", 
                               ["None"] + numeric_cols, key="scatter_size")
    
    if color_option != "None":
        color_col = color_option
    else:
        color_col = None
        
    if size_option != "None":
        size_col = size_option
        size_max = 20
    else:
        size_col = None
        size_max = None
    
    fig = px.scatter(
        df, x=x_col, y=y_col, 
        color=color_col,
        size=size_col, size_max=size_max,
        title=f"Scatter Plot: {x_col} vs {y_col}",
        template="plotly_dark",
        hover_data=df.columns,
        trendline="ols"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)')
    
    st.plotly_chart(fig, use_container_width=True)

def plot_3d_scatter(df, numeric_cols):
    st.subheader("3D Scatter Plot")
    
    if len(numeric_cols) < 3:
        st.warning("Need at least 3 numeric columns for 3D scatter plot.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox("Select X-axis", numeric_cols, key="3d_x")
    
    y_options = numeric_cols.copy()
    default_y_index = 0 if y_options[0] != x_col else min(1, len(y_options)-1)
    
    with col2:
        y_col = st.selectbox("Select Y-axis", y_options, 
                          index=default_y_index, key="3d_y")
    
    z_options = numeric_cols.copy()
    if len(z_options) >= 3:
        for i, col in enumerate(z_options):
            if col != x_col and col != y_col:
                default_z_index = i
                break
        else:
            default_z_index = 0 if z_options[0] != x_col else 1
    else:
        default_z_index = 0
    
    with col3:
        z_col = st.selectbox("Select Z-axis", z_options,
                          index=default_z_index, key="3d_z")
    
    color_option = st.selectbox("Select Color Variable (optional)", 
                              ["None"] + numeric_cols, key="3d_color")
    
    if color_option != "None":
        color_col = color_option
    else:
        color_col = None
    
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col,
        color=color_col,
        title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}",
        template="plotly_dark",
        opacity=0.7
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="white", showbackground=True),
            yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="white", showbackground=True),
            zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="white", showbackground=True),
        ),
        margin=dict(r=0, l=0, b=0, t=30),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    st.subheader("Correlation Heatmap")
    
    corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        text=corr.values.round(6),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hoverinfo='text',
        hovertext=[[f"{y} & {x}: {z:.6f}" for x, z in zip(corr.columns, row)] 
                  for y, row in zip(corr.index, corr.values)],
        colorbar=dict(
            title="Correlation",
            titleside="right"
        )
    ))
    
    fig.update_layout(
        height=700,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title="Correlation Matrix",
        xaxis=dict(
            side="bottom",
            tickmode="array",
            tickvals=list(range(len(corr.columns))),
            ticktext=corr.columns,
            tickangle=-45
        ),
        yaxis=dict(
            autorange="reversed",
            tickmode="array",
            tickvals=list(range(len(corr.index))),
            ticktext=corr.index
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Correlations")
    
    mask = np.triu(np.ones_like(corr), k=1).astype(bool)
    corr_noboth = corr.mask(~mask)
    
    top_pos = corr_noboth.stack().nlargest(5)
    top_neg = corr_noboth.stack().nsmallest(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top Positive Correlations:")
        if not top_pos.empty:
            pos_df = pd.DataFrame({
                'Variables': [f"{i[1]} & {i[0]}" for i in top_pos.index],
                'Correlation': top_pos.values.round(4)
            })
            st.dataframe(pos_df, hide_index=False)
    
    with col2:
        st.write("Top Negative Correlations:")
        if not top_neg.empty:
            neg_df = pd.DataFrame({
                'Variables': [f"{i[1]} & {i[0]}" for i in top_neg.index],
                'Correlation': top_neg.values.round(4)
            })
            st.dataframe(neg_df, hide_index=False)

def plot_distribution(df, numeric_cols):
    st.subheader("Distribution Plot")
    
    col = st.selectbox("Select column:", numeric_cols, key="dist_col")
    
    fig = px.histogram(
        df, x=col,
        marginal=None,
        template="plotly_dark",
        title=f"Distribution of {col}",
        opacity=0.7,
        histnorm="probability density"
    )
    
    kde_resolution = 100
    kde_x = np.linspace(df[col].min(), df[col].max(), kde_resolution)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df[col].dropna())
    kde_y = kde(kde_x)
    
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            line=dict(color='rgba(73, 160, 247, 1)', width=2),
            name='Kernel Density'
        )
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=500,
        bargap=0.05,
        xaxis_title=col,
        yaxis_title="Density"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    stats = df[col].describe()
    st.write("Descriptive Statistics:")
    st.dataframe(stats)

def plot_time_series(df):
    datetime_cols = []
    
    datetime_cols.extend(df.select_dtypes(include=['datetime64']).columns.tolist())
    
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col])
            if col not in datetime_cols:
                datetime_cols.append(col)
        except:
            pass
    
    if not datetime_cols:
        st.warning("No datetime columns found. Please ensure your data contains date/time information.")
        return
    
    st.subheader("Time Series Analysis")
    
    time_col = st.selectbox("Select time/date column:", datetime_cols)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    value_cols = st.multiselect("Select value columns:", numeric_cols)
    
    if not value_cols:
        st.warning("Please select at least one value column for time series analysis.")
        return
    
    ts_df = df.copy()
    try:
        ts_df[time_col] = pd.to_datetime(ts_df[time_col])
    except Exception as e:
        st.error(f"Error converting {time_col} to datetime: {str(e)}")
        return
    
    ts_df = ts_df.sort_values(by=time_col)
    
    fig = go.Figure()
    
    for col in value_cols:
        fig.add_trace(
            go.Scatter(
                x=ts_df[time_col],
                y=ts_df[col],
                name=col,
                mode='lines+markers',
                hovertemplate=
                '%{x}<br>' +
                '%{y}<br>' +
                '<extra></extra>'
            )
        )
    
    fig.update_layout(
        title=f"Time Series: {', '.join(value_cols)} over {time_col}",
        xaxis_title=time_col,
        yaxis_title="Value",
        template="plotly_dark",
        height=500,
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if len(value_cols) > 0:
        st.subheader("Time Series Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev']
        })
        
        for col in value_cols:
            stats_df[col] = [
                ts_df[col].mean(),
                ts_df[col].median(),
                ts_df[col].min(),
                ts_df[col].max(),
                ts_df[col].std()
            ]
        
        st.dataframe(stats_df)

def main():
    st.title("Data Analysis Tool")
    st.write("Upload your data and analyze it using various techniques.")
    
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = process_dataframe(df)
                st.session_state['df'] = df
                st.session_state['filename'] = uploaded_file.name
                st.success(f"Successfully loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        with st.sidebar:
            st.header("Column Selection")
            all_columns = list(df.columns)
            selected_columns = st.multiselect("Select columns to analyze:", all_columns, default=all_columns[:5] if len(all_columns) > 5 else all_columns)
            
            if selected_columns:
                df_selected = df[selected_columns]
                st.session_state['df_selected'] = df_selected
            else:
                st.warning("Please select at least one column.")
        
        if 'df_selected' in st.session_state:
            df_selected = st.session_state['df_selected']
            
            
            tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualization", "Statistical Analysis"])
            
            with tab1:
                st.write(f"File: {st.session_state['filename']} | Shape: {df_selected.shape[0]} rows × {df_selected.shape[1]} columns")
                
                st.subheader("Data Preview")
                st.dataframe(df_selected.head(10))
                
                analyze_data_quality(df_selected)
            
            with tab2:
                st.header("Data Visualization") 
                numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(numeric_cols) < 1:
                    st.warning("No numeric columns available for visualization.")
                else:
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        ["Time Series Analysis", "Correlation Analysis", "Distribution Analysis", 
                         "Scatter Plot", "Box Plot", "3D Scatter Plot"]
                    )
                    
                    if viz_type == "Time Series Analysis":
                        plot_time_series(df_selected)
                    
                    elif viz_type == "Correlation Analysis":
                        plot_correlation_heatmap(df_selected, numeric_cols)
                    
                    elif viz_type == "Distribution Analysis":
                        plot_distribution(df_selected, numeric_cols)
                    
                    elif viz_type == "Scatter Plot":
                        plot_scatter(df_selected, numeric_cols)
                    
                    elif viz_type == "Box Plot":
                        st.subheader("Box Plot")
                        y_col = st.selectbox("Select column for analysis:", numeric_cols)
                        
                        cat_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        if cat_cols:
                            x_col = st.selectbox("Select category for grouping (optional):", ["None"] + cat_cols)
                            
                            if x_col != "None":
                                fig = px.box(df_selected, x=x_col, y=y_col, template="plotly_dark")
                            else:
                                fig = px.box(df_selected, y=y_col, template="plotly_dark")
                        else:
                            fig = px.box(df_selected, y=y_col, template="plotly_dark")
                        
                        fig.update_layout(
                            height=500,
                            plot_bgcolor='rgba(17, 17, 17, 0.8)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "3D Scatter Plot":
                        plot_3d_scatter(df_selected, numeric_cols)
            
            with tab3:
                st.header("Statistical Analysis")
                
                numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if len(numeric_cols) > 0:
                    st.subheader("Descriptive Statistics")
                    st.dataframe(df_selected[numeric_cols].describe())
                    
                    if len(numeric_cols) >= 2:
                        st.subheader("Correlation Analysis")
                        corr = df_selected[numeric_cols].corr()
                        
                        st.write("Top positive correlations:")
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        corr_pairs = corr.mask(mask).stack().sort_values(ascending=False)
                        
                        if not corr_pairs.empty:
                            top_corr = pd.DataFrame({
                                'Pair': [f"{i[0]} & {i[1]}" for i in corr_pairs.index],
                                'Correlation': corr_pairs.values
                            }).head(5)
                            
                            st.dataframe(top_corr)
                else:
                    st.warning("No numeric columns available for statistical analysis.")
    else:
        st.info("Please upload a file from the sidebar to begin.")
        st.markdown("")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or uploading a different file.") 