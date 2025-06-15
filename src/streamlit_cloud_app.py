import streamlit as st

st.set_page_config(
    page_title="Data Analysis Tool",
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from plotnine import (ggplot, aes, geom_point, geom_line, geom_bar, geom_histogram, 
                         geom_boxplot, geom_smooth, geom_violin, scale_x_log10, scale_x_sqrt,
                         scale_y_log10, scale_y_sqrt, facet_wrap, facet_grid, theme_minimal, 
                         labs, theme, element_text)
    PLOTNINE_AVAILABLE = True
except ImportError:
    PLOTNINE_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def process_dataframe(df):
    """Process dataframe to make it Arrow-compatible"""
    
    df_processed = df.copy()
    
    float32_cols = df_processed.select_dtypes(include=['float32']).columns
    if not float32_cols.empty:
        df_processed[float32_cols] = df_processed[float32_cols].astype('float64')
    
    datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
    if not datetime_cols.empty:
        df_processed[datetime_cols] = df_processed[datetime_cols].astype(str)
    
    object_cols = df_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            if pd.to_datetime(df_processed[col], errors='coerce').notna().all():
                df_processed[col] = df_processed[col].astype(str)
        except:
            pass
            
    return df_processed

def analyze_data_quality(df):
    st.header("Data Types and Quality")
    
    dtype_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    dtype_df = pd.DataFrame(list(dtype_dict.items()), columns=['Column', 'Data Type'])
    
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
    
    use_custom_color = st.checkbox("Use custom color", value=False, key="scatter_use_custom")
    if use_custom_color:
        point_color = st.color_picker("Point color:", "#636EFA")
        trendline_color = st.color_picker("Trendline color:", "#FF6692")
    else:
        point_color = None
        trendline_color = None
    
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
    
    if color_col is None and use_custom_color:
        fig = px.scatter(
            df, x=x_col, y=y_col, 
            size=size_col, size_max=size_max,
            title=f"Scatter Plot: {x_col} vs {y_col}",
            template="plotly_dark",
            hover_data=df.columns,
            trendline="ols",
            color_discrete_sequence=[point_color]
        )
        
        for trace in fig.data:
            if trace.mode == 'lines':
                trace.line.color = trendline_color
    else:
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
    
    use_custom_colors = st.checkbox("Use custom colors", value=False, key="3d_custom_colors")
    if use_custom_colors:
        if color_option == "None":
            marker_color = st.color_picker("Marker color:", "#636EFA")
        else:
            col1, col2 = st.columns(2)
            with col1:
                low_color = st.color_picker("Low value color:", "#FF0000")
            with col2:
                high_color = st.color_picker("High value color:", "#0000FF")
            color_scale = [[0, low_color], [1, high_color]]
    else:
        marker_color = "#636EFA"
        color_scale = "Viridis"
    
    if color_option != "None":
        color_col = color_option
    else:
        color_col = None
    
    if color_col is None:
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col,
            title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}",
            template="plotly_dark",
            opacity=0.7,
            color_discrete_sequence=[marker_color] if use_custom_colors else None
        )
    else:
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col,
            color=color_col,
            title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}",
            template="plotly_dark",
            opacity=0.7,
            color_continuous_scale=color_scale if use_custom_colors else None
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

@st.cache_data(show_spinner=False)
def cached_correlation_matrix(df, numeric_cols):
    """Cached computation of correlation matrix"""
    return df[numeric_cols].corr()

@st.cache_data(show_spinner=False)
def cached_describe(df, numeric_cols):
    """Cached computation of descriptive statistics"""
    return df[numeric_cols].describe()

def plot_correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    st.subheader("Correlation Heatmap")
    
    if 'enable_caching' in st.session_state and st.session_state.get('enable_caching', False) and 'cached_corr' in st.session_state and set(numeric_cols).issubset(st.session_state['cached_corr'].columns):
        corr = st.session_state['cached_corr'].loc[numeric_cols, numeric_cols]
    else:
        corr = cached_correlation_matrix(df, numeric_cols)

    if len(numeric_cols) > 20:
        show_all = st.checkbox("Show all columns in heatmap (may be slow)", value=False)
        if not show_all:
            n_cols = st.slider("Number of columns to display:", 5, min(20, len(numeric_cols)), 10)
            
            var_cols = df[numeric_cols].var().sort_values(ascending=False).index[:n_cols].tolist()
            corr = corr.loc[var_cols, var_cols]
            st.info(f"Showing top {n_cols} columns by variance")

    use_custom_colors = st.checkbox("Use custom color scale", value=False, key="corr_use_custom")
    if use_custom_colors:
        col1, col2, col3 = st.columns(3)
        with col1:
            neg_color = st.color_picker("Negative correlation color:", "#0000FF")
        with col2:
            mid_color = st.color_picker("Zero correlation color:", "#FFFFFF")
        with col3:
            pos_color = st.color_picker("Positive correlation color:", "#FF0000")
        
        colorscale = [[0, neg_color], [0.5, mid_color], [1, pos_color]]
    else:
        colorscale = 'RdBu_r'

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=colorscale,
        text=corr.values.round(6),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_traces(
        hoverinfo='text',
        hovertext=[[f"{y} & {x}: {z:.2f}" for x, z in zip(corr.columns, row)] 
                  for y, row in zip(corr.index, corr.values)]
    )
    
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
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
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
    
    col1, col2 = st.columns(2)
    with col1:
        hist_color = st.color_picker("Histogram color:", "#636EFA")
    with col2:
        line_color = st.color_picker("KDE line color:", "#19D3F3")
    
    fig = px.histogram(
        df, x=col,
        marginal=None,
        template="plotly_dark",
        title=f"Distribution of {col}",
        opacity=0.7,
        histnorm="probability density",
        color_discrete_sequence=[hist_color]
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
            line=dict(color=line_color, width=2),
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
    stats_df = pd.DataFrame({
        'Metric': stats.index,
        'Value': stats.values
    })
    st.dataframe(stats_df)

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

def plot_heatmap(df, numeric_cols):
    st.subheader("Heatmap")
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for heatmap visualization.")
        return
    
    selected_cols = st.multiselect("Select columns for heatmap:", numeric_cols, 
                                 default=numeric_cols[:min(5, len(numeric_cols))])
    
    if not selected_cols or len(selected_cols) < 2:
        st.warning("Please select at least 2 columns for the heatmap.")
        return
    
    heatmap_data = df[selected_cols].copy()
    
    normalize = st.checkbox("Normalize data", value=True)
    if normalize:
        for col in selected_cols:
            heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
    
    use_custom_colors = st.checkbox("Use custom color scale", value=False)
    if use_custom_colors:
        col1, col2 = st.columns(2)
        with col1:
            low_color = st.color_picker("Low value color:", "#440154")
        with col2:
            high_color = st.color_picker("High value color:", "#FDE725")
        
        color_scale = [[0, low_color], [1, high_color]]
    else:
        color_scale = 'viridis'
    
    fig = px.imshow(
        heatmap_data.values[:100],
        x=selected_cols,
        y=heatmap_data.index[:100],
        color_continuous_scale=color_scale,
        title="Data Heatmap (First 100 rows)",
        template="plotly_dark"
    )
    
    fig.update_layout(
        height=700,
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_parallel_coordinates(df, numeric_cols):
    st.subheader("Parallel Coordinates Plot")
    
    if len(numeric_cols) < 3:
        st.warning("Need at least 3 numeric columns for parallel coordinates visualization.")
        return
    
    selected_cols = st.multiselect("Select columns for parallel coordinates:", numeric_cols, 
                                  default=numeric_cols[:min(5, len(numeric_cols))])
    
    if len(selected_cols) < 3:
        st.warning("Please select at least 3 columns for the parallel coordinates plot.")
        return
    
    color_by = st.selectbox("Color by:", ["None"] + numeric_cols)
    
    use_custom_colors = st.checkbox("Use custom color scale", value=False, key="parallel_custom_color")
    if use_custom_colors:
        col1, col2 = st.columns(2)
        with col1:
            low_color = st.color_picker("Low value color:", "#FF0000")
        with col2:
            high_color = st.color_picker("High value color:", "#0000FF")
        color_scale = [[0, low_color], [1, high_color]]
    else:
        color_scale = 'viridis'
    
    if color_by != "None":
        fig = px.parallel_coordinates(
            df, 
            dimensions=selected_cols,
            color=df[color_by],
            color_continuous_scale=color_scale,
            title="Parallel Coordinates Plot",
            template="plotly_dark"
        )
    else:
        if use_custom_colors:
            line_color = st.color_picker("Line color:", "#636EFA")
        else:
            line_color = '#636EFA'
            
        fig = px.parallel_coordinates(
            df, 
            dimensions=selected_cols,
            title="Parallel Coordinates Plot",
            template="plotly_dark",
            color_continuous_scale=color_scale
        )
        
        if color_by == "None":
            fig.update_traces(line_color=line_color)
    
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_violin(df, numeric_cols):
    st.subheader("Violin Plot")
    
    y_col = st.selectbox("Select numerical column:", numeric_cols, key="violin_y")
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        x_col = st.selectbox("Group by category (optional):", ["None"] + cat_cols, key="violin_x")
        
        if x_col != "None":
            value_counts = df[x_col].value_counts()
            if len(value_counts) > 10:
                top_categories = value_counts.nlargest(10).index.tolist()
                use_top_categories = st.checkbox("Use only top 10 categories", value=True)
                
                if use_top_categories:
                    filtered_df = df[df[x_col].isin(top_categories)]
                    st.info(f"Showing only top 10 categories out of {len(value_counts)} total categories.")
                else:
                    filtered_df = df
            else:
                filtered_df = df
                
            fig = px.violin(
                filtered_df, 
                x=x_col, 
                y=y_col, 
                box=True,
                points="all",
                template="plotly_dark",
                title=f"Violin Plot: {y_col} by {x_col}"
            )
        else:
            fig = px.violin(
                df, 
                y=y_col, 
                box=True, 
                points="all",
                template="plotly_dark",
                title=f"Violin Plot: {y_col}"
            )
    else:
        fig = px.violin(
            df, 
            y=y_col, 
            box=True, 
            points="all",
            template="plotly_dark",
            title=f"Violin Plot: {y_col}"
        )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_visualization(df, numeric_cols):
    st.subheader("Cluster Visualization")
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering visualization.")
        return
    
    selected_cols = st.multiselect("Select columns for clustering:", numeric_cols, 
                                  default=numeric_cols[:min(5, len(numeric_cols))])
    
    if not selected_cols or len(selected_cols) < 2:
        st.warning("Please select at least 2 columns for clustering visualization.")
        return
    
    clustering_tab, custom_color_tab = st.tabs(["Clustering", "Custom Coloring"])
    
    with clustering_tab:
        cluster_algo = st.selectbox("Select clustering algorithm:", 
                                   ["None", "K-Means", "DBSCAN", "Hierarchical (Agglomerative)"])
        
        cluster_params = {}
        
        if cluster_algo == "K-Means":
            cluster_params['n_clusters'] = st.slider("Number of clusters (K):", 2, 20, 3)
            cluster_params['random_state'] = 42
            
        elif cluster_algo == "DBSCAN":
            cluster_params['eps'] = st.slider("Epsilon (neighborhood distance):", 0.01, 2.0, 0.5)
            cluster_params['min_samples'] = st.slider("Minimum samples in neighborhood:", 2, 20, 5)
            
        elif cluster_algo == "Hierarchical (Agglomerative)":
            cluster_params['n_clusters'] = st.slider("Number of clusters:", 2, 20, 3)
            cluster_params['linkage'] = st.selectbox("Linkage criterion:", 
                                                    ["ward", "complete", "average", "single"])
        
        use_custom_cluster_colors = st.checkbox("Use custom cluster colors", value=False)
        if use_custom_cluster_colors:
            st.info("You can customize up to 5 cluster colors. Additional clusters will use default colors.")
            col1, col2, col3 = st.columns(3)
            with col1:
                cluster_colors = {
                    0: st.color_picker("Cluster 0 color:", "#636EFA"),
                    1: st.color_picker("Cluster 1 color:", "#EF553B")
                }
            with col2:
                cluster_colors.update({
                    2: st.color_picker("Cluster 2 color:", "#00CC96"),
                    3: st.color_picker("Cluster 3 color:", "#AB63FA")
                })
            with col3:
                cluster_colors.update({
                    4: st.color_picker("Cluster 4 color:", "#FFA15A")
                })
    
    with custom_color_tab:
        color_col = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist())
        
        use_custom_color_scale = st.checkbox("Use custom color scale for continuous variables", value=False)
        if use_custom_color_scale and color_col != "None":
            col1, col2 = st.columns(2)
            with col1:
                low_color = st.color_picker("Low value color:", "#440154")
            with col2:
                high_color = st.color_picker("High value color:", "#0000FF")
            color_scale = [[0, low_color], [1, high_color]]
        else:
            color_scale = 'viridis'
    
    st.subheader("Dimension Reduction")
    dim_reduction = st.selectbox("Select dimension reduction method:", 
                               ["None", "PCA 2D", "PCA 3D", "t-SNE 2D", "t-SNE 3D"])
    
    if dim_reduction == "None":
        view_type = st.radio("View type:", ["2D", "3D"], horizontal=True)
        if view_type == "2D":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis column:", selected_cols, key="raw_x")
            with col2:
                y_options = [col for col in selected_cols if col != x_axis]
                if y_options:
                    y_axis = st.selectbox("Y-axis column:", y_options, key="raw_y")
                else:
                    y_axis = x_axis
                    st.warning("Not enough columns selected. Using same column for both axes.")
        else:  # 3D
            if len(selected_cols) < 3:
                st.warning("Need at least 3 selected columns for 3D view. Switching to 2D.")
                view_type = "2D"
                x_axis = selected_cols[0]
                y_axis = selected_cols[1] if len(selected_cols) > 1 else selected_cols[0]
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_axis = st.selectbox("X-axis column:", selected_cols, key="raw_x_3d")
                with col2:
                    y_options = [col for col in selected_cols if col != x_axis]
                    y_axis = st.selectbox("Y-axis column:", y_options, key="raw_y_3d")
                with col3:
                    z_options = [col for col in selected_cols if col != x_axis and col != y_axis]
                    z_axis = st.selectbox("Z-axis column:", z_options, key="raw_z_3d")
    
    X = df[selected_cols].dropna()
    
    if len(X) < 2:
        st.error("Not enough data points after removing missing values.")
        return
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cluster_labels = None
    centroids = None
    cluster_stats = None
    if cluster_algo != "None" and clustering_tab.selected:
        try:
            with st.spinner(f"Computing {cluster_algo} clustering..."):
                if cluster_algo == "K-Means":
                    from sklearn.cluster import KMeans
                    model = KMeans(**cluster_params)
                    cluster_labels = model.fit_predict(X_scaled)
                    centroids = model.cluster_centers_
                    
                elif cluster_algo == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN(**cluster_params)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                elif cluster_algo == "Hierarchical (Agglomerative)":
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(**cluster_params)
                    cluster_labels = model.fit_predict(X_scaled)
            
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)
            st.write(f"Total clusters: {n_clusters}")
            
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
            cluster_labels = None
    
    if dim_reduction == "None":
        indices = []
        for col in selected_cols:
            indices.append(selected_cols.index(col))
            
        if view_type == "3D":
            x_idx = selected_cols.index(x_axis)
            y_idx = selected_cols.index(y_axis)
            z_idx = selected_cols.index(z_axis)
            
            X_reduced = np.column_stack((
                X_scaled[:, x_idx],
                X_scaled[:, y_idx],
                X_scaled[:, z_idx]
            ))
            n_components = 3
            reducer_name = f"Raw Data ({x_axis}, {y_axis}, {z_axis})"
        else:  # 2D
            x_idx = selected_cols.index(x_axis)
            y_idx = selected_cols.index(y_axis)
            
            X_reduced = np.column_stack((
                X_scaled[:, x_idx],
                X_scaled[:, y_idx]
            ))
            n_components = 2
            reducer_name = f"Raw Data ({x_axis}, {y_axis})"
    elif "PCA" in dim_reduction:
        from sklearn.decomposition import PCA
        n_components = 3 if "3D" in dim_reduction else 2
        reducer = PCA(n_components=n_components)
        reducer_name = "PCA"
        with st.spinner(f"Computing {reducer_name}..."):
            X_reduced = reducer.fit_transform(X_scaled)
    else:  # t-SNE
        from sklearn.manifold import TSNE
        n_components = 3 if "3D" in dim_reduction else 2
        perplexity = min(30, len(X) - 1) if len(X) > 30 else max(5, len(X) // 5)
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        reducer_name = "t-SNE"
        with st.spinner(f"Computing {reducer_name}..."):
            X_reduced = reducer.fit_transform(X_scaled)
    
    if n_components == 2:
        plot_df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1]
        })
    else:  # 3D
        plot_df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1],
            'Component 3': X_reduced[:, 2]
        })
    
    if dim_reduction == "None":
        if n_components == 2:
            x_label = x_axis
            y_label = y_axis
        else:  # 3D
            x_label = x_axis
            y_label = y_axis
            z_label = z_axis
    else:
        if n_components == 2:
            x_label = f"{reducer_name} Component 1"
            y_label = f"{reducer_name} Component 2"
        else:  # 3D
            x_label = f"{reducer_name} Component 1"
            y_label = f"{reducer_name} Component 2"
            z_label = f"{reducer_name} Component 3"
    
    if cluster_labels is not None and clustering_tab.selected:
        plot_df['Cluster'] = [f"Cluster {label}" for label in cluster_labels]
        color_column = 'Cluster'
    elif color_col != "None" and custom_color_tab.selected:
        if len(df[color_col].dropna()) == len(X):
            plot_df['Color'] = df[color_col].values[X.index]
            color_column = 'Color'
        else:
            st.warning(f"Cannot use {color_col} for coloring due to missing values. Using default colors.")
            color_column = None
    else:
        color_column = None
    
    if n_components == 2:  # 2D plot
        if color_column:
            if color_column == 'Cluster' and clustering_tab.selected and use_custom_cluster_colors:
                color_map = {}
                for i, cluster_name in enumerate(plot_df['Cluster'].unique()):
                    cluster_idx = int(cluster_name.split(" ")[1])
                    if cluster_idx in cluster_colors:
                        color_map[cluster_name] = cluster_colors[cluster_idx]
                
                fig = px.scatter(
                    plot_df,
                    x='Component 1',
                    y='Component 2',
                    color=color_column,
                    color_discrete_map=color_map if color_map else None,
                    title=f"{reducer_name} 2D Projection",
                    template="plotly_dark",
                    labels={'Component 1': x_label, 
                            'Component 2': y_label}
                )
            else:
                if custom_color_tab.selected and use_custom_color_scale and color_col != "None":
                    fig = px.scatter(
                        plot_df,
                        x='Component 1',
                        y='Component 2',
                        color=color_column,
                        color_continuous_scale=color_scale,
                        title=f"{reducer_name} 2D Projection",
                        template="plotly_dark",
                        labels={'Component 1': x_label, 
                                'Component 2': y_label}
                    )
                else:
                    fig = px.scatter(
                        plot_df,
                        x='Component 1',
                        y='Component 2',
                        color=color_column,
                        title=f"{reducer_name} 2D Projection",
                        template="plotly_dark",
                        labels={'Component 1': x_label, 
                                'Component 2': y_label}
                    )
        else:
            point_color = "#636EFA"
            if not color_column:
                use_solid_color = st.checkbox("Use custom point color", value=False)
                if use_solid_color:
                    point_color = st.color_picker("Point color:", point_color)
            
            fig = px.scatter(
                plot_df,
                x='Component 1',
                y='Component 2',
                title=f"{reducer_name} 2D Projection",
                template="plotly_dark",
                labels={'Component 1': x_label, 
                        'Component 2': y_label},
                color_discrete_sequence=[point_color]
            )
    else:  # 3D plot
        if color_column:
            if color_column == 'Cluster' and clustering_tab.selected and use_custom_cluster_colors:
                color_map = {}
                for i, cluster_name in enumerate(plot_df['Cluster'].unique()):
                    cluster_idx = int(cluster_name.split(" ")[1])
                    if cluster_idx in cluster_colors:
                        color_map[cluster_name] = cluster_colors[cluster_idx]
                
                fig = px.scatter_3d(
                    plot_df,
                    x='Component 1',
                    y='Component 2',
                    z='Component 3',
                    color=color_column,
                    color_discrete_map=color_map if color_map else None,
                    title=f"{reducer_name} 3D Projection",
                    template="plotly_dark",
                    labels={'Component 1': x_label, 
                            'Component 2': y_label,
                            'Component 3': z_label}
                )
            else:
                if custom_color_tab.selected and use_custom_color_scale and color_col != "None":
                    fig = px.scatter_3d(
                        plot_df,
                        x='Component 1',
                        y='Component 2',
                        z='Component 3',
                        color=color_column,
                        color_continuous_scale=color_scale,
                        title=f"{reducer_name} 3D Projection",
                        template="plotly_dark",
                        labels={'Component 1': x_label, 
                                'Component 2': y_label,
                                'Component 3': z_label}
                    )
                else:
                    fig = px.scatter_3d(
                        plot_df,
                        x='Component 1',
                        y='Component 2',
                        z='Component 3',
                        color=color_column,
                        title=f"{reducer_name} 3D Projection",
                        template="plotly_dark",
                        labels={'Component 1': x_label, 
                                'Component 2': y_label,
                                'Component 3': z_label}
                    )
        else:
            point_color = "#636EFA"
            if not color_column:
                use_solid_color = st.checkbox("Use custom point color (3D)", value=False)
                if use_solid_color:
                    point_color = st.color_picker("Point color (3D):", point_color)
            
            fig = px.scatter_3d(
                plot_df,
                x='Component 1',
                y='Component 2',
                z='Component 3',
                title=f"{reducer_name} 3D Projection",
                template="plotly_dark",
                labels={'Component 1': x_label, 
                        'Component 2': y_label,
                        'Component 3': z_label},
                color_discrete_sequence=[point_color]
            )
    
    fig.update_layout(
        height=700,
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    
    if color_column:
        hover_data = {}
        for col in selected_cols:
            hover_data[col] = X[col].values
        fig.update_traces(hovertemplate='<br>'.join([
            'Component 1: %{x}',
            'Component 2: %{y}',
            *[f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(selected_cols)]
        ]))
        fig.update_traces(customdata=np.column_stack([X[col].values for col in selected_cols]))
    
    st.plotly_chart(fig, use_container_width=True)
    
    if dim_reduction != "None" and "PCA" in dim_reduction:
        explained_variance = reducer.explained_variance_ratio_
        total_explained_variance = explained_variance.sum() * 100
        
        st.write(f"Explained variance ratio per component:")
        for i, variance in enumerate(explained_variance):
            st.write(f"Component {i+1}: {variance:.4f} ({variance*100:.2f}%)")
        
        st.write(f"Total explained variance: {total_explained_variance:.2f}%")
    
    if cluster_labels is not None and clustering_tab.selected:
        unique_clusters = np.unique(cluster_labels)
        cluster_counts = {}
        for cluster in unique_clusters:
            count = np.sum(cluster_labels == cluster)
            cluster_counts[f"Cluster {cluster}"] = count
        
        st.header("Cluster Information")
        
        cluster_df = pd.DataFrame({
            'Cluster': list(cluster_counts.keys()),
            'Count': list(cluster_counts.values()),
            'Percentage': [f"{(count/len(cluster_labels)*100):.2f}%" for count in cluster_counts.values()]
        })
        
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write("## Cluster Summary")
            st.dataframe(cluster_df)
        
        with col2:
            if use_custom_cluster_colors:
                color_map = {}
                for i, cluster_name in enumerate(cluster_df['Cluster']):
                    cluster_idx = int(cluster_name.split(" ")[1])
                    if cluster_idx in cluster_colors:
                        color_map[cluster_name] = cluster_colors[cluster_idx]
                
                fig_pie = px.pie(
                    cluster_df, 
                    values='Count', 
                    names='Cluster',
                    title="Cluster Distribution",
                    template="plotly_dark",
                    color='Cluster',
                    color_discrete_map=color_map if color_map else None
                )
            else:
                fig_pie = px.pie(
                    cluster_df, 
                    values='Count', 
                    names='Cluster',
                    title="Cluster Distribution",
                    template="plotly_dark"
                )
            
            fig_pie.update_layout(
                plot_bgcolor='rgba(17, 17, 17, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        if cluster_algo == "K-Means" and centroids is not None:
            st.write("## Cluster Centroids")
            
            centroids_original = scaler.inverse_transform(centroids)
            
            centroid_df = pd.DataFrame(
                centroids_original,
                columns=selected_cols,
                index=[f"Cluster {i}" for i in range(len(centroids))]
            )
            
            st.dataframe(centroid_df)
            
            centroid_melted = centroid_df.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
            
            if use_custom_cluster_colors:
                color_map = {}
                for i in range(len(centroids)):
                    cluster_name = f"Cluster {i}"
                    if i in cluster_colors:
                        color_map[cluster_name] = cluster_colors[i]
                
                fig_bar = px.bar(
                    centroid_melted, 
                    x='Feature', 
                    y='Value', 
                    color='index',
                    barmode='group',
                    title="Centroid Values by Feature",
                    template="plotly_dark",
                    color_discrete_map=color_map if color_map else None
                )
            else:
                fig_bar = px.bar(
                    centroid_melted, 
                    x='Feature', 
                    y='Value', 
                    color='index',
                    barmode='group',
                    title="Centroid Values by Feature",
                    template="plotly_dark"
                )
            
            fig_bar.update_layout(
                plot_bgcolor='rgba(17, 17, 17, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis_title="Feature",
                yaxis_title="Value",
                legend_title="Cluster"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write("## Detailed Cluster Analysis")
        X_with_clusters = X.copy()
        X_with_clusters['Cluster'] = [f"Cluster {label}" for label in cluster_labels]
        
        cluster_stats_list = []
        for cluster in unique_clusters:
            cluster_data = X_with_clusters[X_with_clusters['Cluster'] == f"Cluster {cluster}"]
            cluster_stats = {}
            for col in selected_cols:
                cluster_stats[f"{col}_mean"] = cluster_data[col].mean()
                cluster_stats[f"{col}_median"] = cluster_data[col].median()
                cluster_stats[f"{col}_std"] = cluster_data[col].std()
            cluster_stats['Cluster'] = f"Cluster {cluster}"
            cluster_stats_list.append(cluster_stats)
        
        if cluster_stats_list:
            cluster_stats_df = pd.DataFrame(cluster_stats_list)
            cluster_stats_df = cluster_stats_df.set_index('Cluster')
            
            stats_by_feature = {}
            for col in selected_cols:
                stats_by_feature[col] = pd.DataFrame({
                    'Mean': cluster_stats_df[f"{col}_mean"],
                    'Median': cluster_stats_df[f"{col}_median"],
                    'Std Dev': cluster_stats_df[f"{col}_std"]
                })
            
            feature_tabs = st.tabs(selected_cols)
            for i, col in enumerate(selected_cols):
                with feature_tabs[i]:
                    st.dataframe(stats_by_feature[col])
                    
                    fig_stat = px.bar(
                        stats_by_feature[col].reset_index(),
                        x='Cluster',
                        y='Mean',
                        title=f"Mean {col} by Cluster",
                        template="plotly_dark",
                        error_y='Std Dev'
                    )
                    fig_stat.update_layout(
                        plot_bgcolor='rgba(17, 17, 17, 0.8)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                    )
                    st.plotly_chart(fig_stat, use_container_width=True)

def ggplot_playground(df):
    """Interactive ggplot code generator - enhanced and educational"""
    
    if not PLOTNINE_AVAILABLE:
        st.error("plotnine is required for the ggplot playground. Please install it with: pip install plotnine")
        st.info("plotnine is the Python implementation of ggplot2's Grammar of Graphics")
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
                box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);">
        <h2 style="color: white; text-align: center; margin: 0; font-weight: 700; font-size: 2rem;">
            Grammar of Graphics Playground
        </h2>
        <p style="color: rgba(255, 255, 255, 0.9); text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Build Interactive Visualizations Using Grammar of Graphics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    if len(all_cols) == 0:
        st.warning("No suitable columns found for plotting!")
        return
    
    with st.expander("Preview Your Data", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Shape:**")
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Numeric:** {len(numeric_cols)}")
            st.write(f"**Categorical:** {len(categorical_cols)}")
        
        with col2:
            st.markdown("**Column Types:**")
            if numeric_cols:
                st.write(f"**Numeric:** {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}")
            if categorical_cols:
                st.write(f"**Categorical:** {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}")
        
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(3), use_container_width=True)
    
    st.subheader("Build Your Plot")
    
    x_default = "None"
    y_default = "None"
    geom_default = 'geom_point'
    color_default = "None"
    
    x_var = st.selectbox("X variable:", ["None"] + all_cols, 
                       index=all_cols.index(x_default) + 1 if x_default in all_cols else 0)
    y_var = st.selectbox("Y variable:", ["None"] + all_cols,
                       index=all_cols.index(y_default) + 1 if y_default in all_cols else 0)
    color_var = st.selectbox("Color by (optional):", ["None"] + all_cols,
                            index=all_cols.index(color_default) + 1 if color_default in all_cols else 0)
    
    geom_options = {
        "geom_point": "Scatter Plot (points)",
        "geom_line": "Line Plot",
        "geom_bar": "Bar Chart",
        "geom_histogram": "Histogram", 
        "geom_boxplot": "Box Plot",
        "geom_violin": "Violin Plot (density distribution)",
        "geom_density": "Density Curve (smooth distribution)",
        "geom_area": "Area Plot (filled line chart)",
        "geom_text": "Text Labels (add labels to points)",
        "geom_jitter": "Jittered Points (scattered categorical data)",
        "geom_tile": "Heatmap Tiles (grid visualization)"
    }
    
    geom_index = 0
    
    geom_display = st.selectbox("Plot type:", list(geom_options.keys()), 
                               format_func=lambda x: geom_options[x],
                               index=geom_index)
    
    geom_help = {
        "geom_point": "Perfect for exploring relationships between two continuous variables",
        "geom_line": "Great for time series or connecting ordered data points",
        "geom_bar": "Ideal for categorical data or counts",
        "geom_histogram": "Shows distribution of a single continuous variable",
        "geom_boxplot": "Compares distributions across categories",
        "geom_violin": "Shows full distribution shape, like boxplot + density",
        "geom_density": "Smooth curve showing data distribution",
        "geom_area": "Filled area under a line, great for cumulative data",
        "geom_text": "Add text labels to data points for annotation",
        "geom_jitter": "Scatter points to avoid overplotting in categorical data",
        "geom_tile": "Create heatmaps and grid-based visualizations"
    }
    st.caption(f"Tip: {geom_help[geom_display]}")
    
    theme_options = {
        "theme_minimal": "Clean and minimal",
        "theme_classic": "Classic with borders", 
        "theme_gray": "Traditional gray background"
    }
    theme_choice = st.selectbox("Theme:", list(theme_options.keys()),
                               format_func=lambda x: theme_options[x])
    
    alpha_val = st.slider("Transparency:", 0.1, 1.0, 0.7, 0.1)
    
    coord_system = "default"
    x_scale = "default"
    y_scale = "default"
    color_scale = "default"
    brewer_choice = None
    cmap_choice = None
    text_var = "None"
    text_size = 10
    facet_type = "None"
    facet_var = "None"
    facet_row = "None"
    facet_col = "None"
    facet_scales = "fixed"
    facet_cols = 2
    add_smooth = False
    smooth_method = "lm"
    smooth_se = True
    stat_transform = None
    
    with st.expander("Advanced Options", expanded=False):
        
        st.markdown("**Coordinate Systems:**")
        coord_options = {
            "default": "Default Cartesian",
            "coord_flip": "Flip X and Y axes"
        }
        coord_system = st.selectbox("Coordinate system:", list(coord_options.keys()),
                                  format_func=lambda x: coord_options[x])
        
        st.markdown("**Scales & Transformations:**")
        col1, col2 = st.columns(2)
        with col1:
            x_scale = st.selectbox("X-axis scale:", 
                                 ["default", "log10", "sqrt", "reverse"],
                                 help="Transform the X-axis scale")
        with col2:
            y_scale = st.selectbox("Y-axis scale:", 
                                 ["default", "log10", "sqrt", "reverse"],
                                 help="Transform the Y-axis scale")
        
        if color_var != "None":
            st.markdown("**Color Scale:**")
            color_scale_options = {
                "default": "Default colors",
                "brewer": "ColorBrewer palette",
                "cmap": "Colormap palette",
                "manual": "Custom colors"
            }
            color_scale = st.selectbox("Color palette:", list(color_scale_options.keys()),
                                     format_func=lambda x: color_scale_options[x])
            
            if color_scale == "brewer":
                brewer_palettes = ["Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Dark2", "Accent"]
                brewer_choice = st.selectbox("ColorBrewer palette:", brewer_palettes)
            elif color_scale == "cmap":
                cmap_palettes = ["viridis", "plasma", "inferno", "magma", "cividis", "tab10", "tab20"]
                cmap_choice = st.selectbox("Colormap palette:", cmap_palettes)
            elif color_scale == "manual":
                st.info("Manual color selection - specify colors in generated code")
        
        if geom_display == "geom_text":
            text_var = st.selectbox("Text label variable:", ["None"] + all_cols,
                                  help="Choose which column to use as text labels")
            text_size = st.slider("Text size:", 6, 20, 10)
        
        st.markdown("**Faceting (Small Multiples):**")
        facet_type = st.radio("Facet type:", ["None", "facet_wrap", "facet_grid"], horizontal=True)
        
        if facet_type != "None":
            if facet_type == "facet_wrap":
                facet_var = st.selectbox("Facet by:", ["None"] + categorical_cols)
                if facet_var != "None":
                    facet_cols = st.slider("Number of columns:", 1, 5, 2)
                    facet_scales = st.selectbox("Scales:", ["fixed", "free", "free_x", "free_y"],
                                              help="fixed: same scales, free: independent scales")
            else:  # facet_grid
                facet_row = st.selectbox("Facet rows by:", ["None"] + categorical_cols)
                facet_col = st.selectbox("Facet columns by:", ["None"] + categorical_cols)
                facet_scales = st.selectbox("Scales:", ["fixed", "free", "free_x", "free_y"])
        
        if geom_display in ["geom_point", "geom_line", "geom_jitter"]:
            add_smooth = st.checkbox("Add trend line (geom_smooth)", value=False)
            if add_smooth:
                smooth_method = st.selectbox("Smooth method:", ["lm", "loess", "gam"],
                                           help="lm: linear, loess: local regression, gam: generalized additive")
                smooth_se = st.checkbox("Show confidence interval", value=True)
            
        if geom_display == "geom_bar":
            stat_transform = st.selectbox("Statistical transform:", 
                                        ["count", "identity"], 
                                        help="count: count observations, identity: use y values as-is")
    
    st.subheader("Your Plot")
    
    
    should_show_plot = True
    auto_message = ""
    
    if geom_display in ["geom_histogram", "geom_density"]:
        should_show_plot = x_var != "None"
        if not should_show_plot:
            auto_message = f"Select an X variable to see your {geom_display.replace('geom_', '')} plot"
    elif geom_display in ["geom_point", "geom_line", "geom_area"]:
        should_show_plot = x_var != "None" and y_var != "None"
        if not should_show_plot:
            if x_var == "None":
                auto_message = "Select X and Y variables to see your scatter/line plot"
            else:
                auto_message = "Select a Y variable to complete your plot"
    elif geom_display == "geom_bar":
        should_show_plot = x_var != "None"
        if not should_show_plot:
            auto_message = "Select an X variable to see your bar chart"
    elif geom_display == "geom_text":
        should_show_plot = x_var != "None" and y_var != "None" and text_var != "None"
        if not should_show_plot:
            if x_var == "None" or y_var == "None":
                auto_message = "Select X and Y variables, then choose a Text variable in Advanced Options"
            else:
                auto_message = "Choose a Text variable in Advanced Options to show labels"
    elif geom_display == "geom_tile":
        should_show_plot = x_var != "None" and y_var != "None"
        if not should_show_plot:
            auto_message = "Select X and Y variables for your heatmap"
    else:
        should_show_plot = x_var != "None"
        if not should_show_plot:
            auto_message = "Select an X variable to start plotting"
    
    if not should_show_plot:
        st.markdown(f"""
        <div style="background-color: #e8f4fd; border: 1px solid #3498db; border-radius: 8px; padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #2980b9; margin-top: 0;">🚀 Ready to Create Your Plot?</h3>
            <p style="margin-bottom: 0; color: #34495e;">
                <strong>Next step:</strong> {auto_message}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            if len(df) > 1000:
                df_plot = df.sample(1000, random_state=42)
                st.info("Using 1000 samples for performance")
            else:
                df_plot = df
            
            aes_args = {}
            if x_var != "None":
                aes_args['x'] = x_var
            if y_var != "None" and geom_display not in ["geom_histogram", "geom_density"]:
                aes_args['y'] = y_var
            if color_var != "None":
                aes_args['color'] = color_var
            
            p = ggplot(df_plot, aes(**aes_args))
            
            if geom_display == "geom_point":
                p += geom_point(alpha=alpha_val, size=2)
            elif geom_display == "geom_line":
                p += geom_line(alpha=alpha_val)
            elif geom_display == "geom_bar":
                if stat_transform == "identity" and y_var != "None":
                    p += geom_bar(alpha=alpha_val, stat="identity")
                else:
                    p += geom_bar(alpha=alpha_val)
            elif geom_display == "geom_histogram":
                p += geom_histogram(alpha=alpha_val, bins=30)
            elif geom_display == "geom_boxplot":
                p += geom_boxplot(alpha=alpha_val)
            elif geom_display == "geom_violin":
                try:
                    from plotnine import geom_violin
                    p += geom_violin(alpha=alpha_val)
                except ImportError:
                    st.error("geom_violin requires plotnine with extended geometries")
                    return
            elif geom_display == "geom_density":
                try:
                    from plotnine import geom_density
                    p += geom_density(alpha=alpha_val)
                except ImportError:
                    st.error("geom_density requires plotnine with extended geometries")
                    return
            elif geom_display == "geom_area":
                try:
                    from plotnine import geom_area
                    p += geom_area(alpha=alpha_val)
                except ImportError:
                    st.error("geom_area requires plotnine with extended geometries")
                    return
            elif geom_display == "geom_text":
                try:
                    from plotnine import geom_text
                    aes_text = aes_args.copy()
                    aes_text['label'] = text_var
                    p = ggplot(df_plot, aes(**aes_text))
                    p += geom_text(alpha=alpha_val, size=text_size)
                except ImportError:
                    st.error("geom_text requires plotnine with extended geometries")
                    return
            elif geom_display == "geom_jitter":
                try:
                    from plotnine import geom_jitter
                    p += geom_jitter(alpha=alpha_val, width=0.2, height=0.2)
                except ImportError:
                    st.error("geom_jitter requires plotnine with extended geometries")
                    return
            elif geom_display == "geom_tile":
                try:
                    from plotnine import geom_tile
                    p += geom_tile(alpha=alpha_val)
                except ImportError:
                    st.error("geom_tile requires plotnine with extended geometries")
                    return
            
            if add_smooth and geom_display in ["geom_point", "geom_line", "geom_jitter"]:
                try:
                    from plotnine import geom_smooth
                    p += geom_smooth(method=smooth_method, se=smooth_se, alpha=0.3)
                except ImportError:
                    st.info("geom_smooth requires additional packages")
            
            if facet_type == "facet_wrap" and facet_var != "None":
                try:
                    from plotnine import facet_wrap
                    p += facet_wrap(f"~{facet_var}", ncol=facet_cols, scales=facet_scales)
                except ImportError:
                    st.info("Faceting requires additional packages")
            elif facet_type == "facet_grid" and (facet_row != "None" or facet_col != "None"):
                try:
                    from plotnine import facet_grid
                    if facet_row != "None" and facet_col != "None":
                        p += facet_grid(f"{facet_row}~{facet_col}", scales=facet_scales)
                    elif facet_row != "None":
                        p += facet_grid(f"{facet_row}~.", scales=facet_scales)
                    elif facet_col != "None":
                        p += facet_grid(f".~{facet_col}", scales=facet_scales)
                except ImportError:
                    st.info("Faceting requires additional packages")
            
            if coord_system == "coord_flip":
                try:
                    from plotnine import coord_flip
                    p += coord_flip()
                except ImportError:
                    st.info("coord_flip requires additional packages")
            
            if x_scale != "default":
                try:
                    if x_scale == "log10":
                        from plotnine import scale_x_log10
                        p += scale_x_log10()
                    elif x_scale == "sqrt":
                        from plotnine import scale_x_sqrt
                        p += scale_x_sqrt()
                    elif x_scale == "reverse":
                        from plotnine import scale_x_reverse
                        p += scale_x_reverse()
                except ImportError:
                    st.info(f"scale_x_{x_scale} requires additional packages")
            
            if y_scale != "default":
                try:
                    if y_scale == "log10":
                        from plotnine import scale_y_log10
                        p += scale_y_log10()
                    elif y_scale == "sqrt":
                        from plotnine import scale_y_sqrt
                        p += scale_y_sqrt()
                    elif y_scale == "reverse":
                        from plotnine import scale_y_reverse
                        p += scale_y_reverse()
                except ImportError:
                    st.info(f"scale_y_{y_scale} requires additional packages")
            
            if color_var != "None" and color_scale != "default":
                try:
                    if color_scale == "brewer":
                        from plotnine import scale_color_brewer
                        p += scale_color_brewer(type="qual", palette=brewer_choice)
                    elif color_scale == "cmap":
                        from plotnine import scale_color_cmap
                        p += scale_color_cmap(cmap_choice)
                except ImportError:
                    st.info("Advanced color scales require additional packages")
            
            if theme_choice == "theme_minimal":
                p += theme_minimal()
            elif theme_choice == "theme_classic":
                try:
                    from plotnine import theme_classic
                    p += theme_classic()
                except:
                    p += theme_minimal()
            else:
                try:
                    from plotnine import theme_gray
                    p += theme_gray()
                except:
                    p += theme_minimal()
            
            try:
                with st.spinner("Generating your plot..."):
                    fig = p.draw()
                    fig.set_size_inches(10, 6)
                    
                st.markdown("""
                <div style="background-color: white; padding: 1.5rem; border-radius: 8px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 1rem 0; min-height: 500px;">
                """, unsafe_allow_html=True)
                
                st.pyplot(fig, use_container_width=True, clear_figure=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.success("Plot generated successfully!")
                
            except Exception as e:
                st.error(f"Plot Error: {str(e)}")
                
                with st.expander("Show Error Details"):
                    import traceback
                    error_details = traceback.format_exc()
                    st.code(error_details)
                    st.write(f"**Variables:** X={x_var}, Y={y_var}, Color={color_var}")
                    st.write(f"**Plot type:** {geom_display}")
                    st.write(f"**Data shape:** {df_plot.shape}")
            
            st.markdown("### Generated Code")
            aes_str = ", ".join([f"{k}='{v}'" for k, v in aes_args.items()])
            
            code_lines = ["from plotnine import *", "", ""]
            
            if geom_display == "geom_bar" and stat_transform == "identity" and y_var != "None":
                code_lines.append(f"     {geom_display}(alpha={alpha_val}, stat='identity') +")
            elif geom_display == "geom_text":
                code_lines.append(f"     {geom_display}(alpha={alpha_val}, size={text_size}) +")
            elif geom_display == "geom_jitter":
                code_lines.append(f"     {geom_display}(alpha={alpha_val}, width=0.2, height=0.2) +")
            elif geom_display == "geom_histogram":
                code_lines.append(f"     {geom_display}(alpha={alpha_val}, bins=30) +")
            else:
                code_lines.append(f"     {geom_display}(alpha={alpha_val}) +")
            
            if add_smooth and geom_display in ["geom_point", "geom_line", "geom_jitter"]:
                code_lines.append(f"     geom_smooth(method='{smooth_method}', se={smooth_se}, alpha=0.3) +")
            
            if coord_system != "default":
                code_lines.append(f"     {coord_system}() +")
                
            if x_scale != "default":
                code_lines.append(f"     scale_x_{x_scale}() +")
            if y_scale != "default":
                code_lines.append(f"     scale_y_{y_scale}() +")
            
            if color_var != "None" and color_scale != "default":
                if color_scale == "brewer":
                    code_lines.append(f"     scale_color_brewer(type='qual', palette='{brewer_choice}') +")
                elif color_scale == "cmap":
                    code_lines.append(f"     scale_color_cmap('{cmap_choice}') +")
            
            code_lines.append(f"     {theme_choice}()")
            
            if facet_type == "facet_wrap" and facet_var != "None":
                code_lines.append(f"     + facet_wrap('~{facet_var}', ncol={facet_cols}, scales='{facet_scales}')")
            elif facet_type == "facet_grid" and (facet_row != "None" or facet_col != "None"):
                if facet_row != "None" and facet_col != "None":
                    code_lines.append(f"     + facet_grid('{facet_row}~{facet_col}', scales='{facet_scales}')")
                elif facet_row != "None":
                    code_lines.append(f"     + facet_grid('{facet_row}~.', scales='{facet_scales}')")
                elif facet_col != "None":
                    code_lines.append(f"     + facet_grid('.~{facet_col}', scales='{facet_scales}')")
            
            code_lines.append(")")
            code_lines.append("")
            code_lines.append("print(p)")
            
            code = "\n".join(code_lines)
            st.code(code, language="python")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure plotnine is installed and try simpler variable combinations")
    
    with st.expander("Learn Grammar of Graphics", expanded=False):
        
        learn_tab1, learn_tab2, learn_tab3 = st.tabs(["Concepts", "Components", "Examples"])
        
        with learn_tab1:
            st.markdown("""
            
            **Grammar of Graphics** is a systematic approach to creating data visualizations by breaking them into components:
            
            - **Declarative**: You describe *what* you want, not *how* to draw it
            - **Modular**: Build complex plots from simple, reusable components  
            - **Flexible**: Easy to modify and extend existing plots
            - **Consistent**: Same logic applies to all plot types
            
            **Why ggplot?**
            - Beautiful defaults that follow visualization best practices
            - Powerful for exploratory data analysis
            - Consistent syntax across different plot types
            - Easy to create publication-ready graphics
            """)
        
        with learn_tab2:
            st.markdown("""
            
            Every ggplot has these building blocks:
            
            1. **Data**: Your dataset (DataFrame)
            2. **Aesthetics (`aes`)**: Map variables to visual properties
               - `x`, `y`: Position
               - `color`: Color of points/lines
               - `size`: Size of points
               - `alpha`: Transparency
               - `label`: Text labels (for geom_text)
            3. **Geometries (`geom_*`)**: Visual elements
               - `geom_point()`: Scatter plots
               - `geom_line()`: Line charts  
               - `geom_bar()`: Bar charts
               - `geom_histogram()`: Histograms
               - `geom_boxplot()`: Box plots
               - `geom_violin()`: Violin plots (density + box)
               - `geom_density()`: Smooth density curves
               - `geom_area()`: Filled area plots
               - `geom_text()`: Text annotations
               - `geom_jitter()`: Jittered scatter plots
               - `geom_tile()`: Heatmap tiles
            4. **Coordinate Systems**: Transform the plotting space
               - `coord_flip()`: Flip X and Y axes
               - `coord_polar()`: Polar coordinates
            5. **Scales**: Control axis and color mappings
               - `scale_x_log10()`: Logarithmic X-axis
               - `scale_color_viridis()`: Viridis color palette
               - `scale_color_brewer()`: ColorBrewer palettes
            6. **Faceting**: Create small multiples
               - `facet_wrap()`: Wrap plots in a grid
               - `facet_grid()`: Arrange in rows and columns
            7. **Themes**: Overall appearance and styling
            """)
            
        with learn_tab3:
            st.markdown("""
            
            **Basic Scatter Plot:**
            ```python
            ggplot(df, aes(x='age', y='salary')) + geom_point()
            ```
            
            **Add Color by Category:**
            ```python
            ggplot(df, aes(x='age', y='salary', color='department')) + geom_point()
            ```
            
            **Horizontal Bar Chart (Coordinate Flip):**
            ```python
            (ggplot(df, aes(x='department', y='salary')) +
             geom_bar(stat='identity') +
             coord_flip())
            ```
            
            **Logarithmic Scale:**
            ```python
            (ggplot(df, aes(x='age', y='salary')) +
             geom_point() +
             scale_y_log10())
            ```
            
            **Text Annotations:**
            ```python
            (ggplot(df, aes(x='age', y='salary', label='name')) +
             geom_point() +
             geom_text())
            ```
            
            **Violin Plot with Custom Colors:**
            ```python
            (ggplot(df, aes(x='department', y='salary')) +
             geom_violin() +
             scale_color_viridis_d())
            ```
            
            **Advanced Grid Faceting:**
            ```python
            (ggplot(df, aes(x='age', y='salary')) +
             geom_point() +
             facet_grid('department~gender') +
             theme_minimal())
            ```
            
            **Multiple Layers:**
            ```python
            (ggplot(df, aes(x='age', y='salary')) +
             geom_point() +
             geom_smooth(method='lm') +
             theme_minimal())
            ```
            
            **Combined Advanced Features:**
            ```python
            (ggplot(df, aes(x='age', y='salary', color='department')) +
             geom_point(alpha=0.7) +
             geom_smooth(method='lm', se=True) +
             facet_wrap('~department') +
             scale_color_brewer(type='qual', palette='Set1') +
             theme_minimal())
            ```
            
            **Try these patterns in the playground above!**
            """)
    
    with st.expander("Pro Tips & Best Practices", expanded=False):
        st.markdown("""
        
        - **Color wisely**: Use color to highlight important patterns, not just for decoration
        - **Mind your scales**: Log scales can reveal hidden patterns in skewed data
        - **Less is more**: Don't overcrowd plots with too many variables
        - **Think about your audience**: Will they view this on mobile or desktop?
        - **Accessibility**: Use colorblind-friendly palettes when possible
        
        - Try the **faceting** functions (`facet_wrap`, `facet_grid`) for small multiples
        - Explore **coordinate systems** (`coord_flip`, `coord_polar`) for different perspectives
        - Learn about **statistical layers** (`stat_summary`, `stat_smooth`) for data summaries
        - Master **scales** (`scale_x_log10`, `scale_color_manual`) for fine control
        """)
    
    with st.expander("Export", expanded=False):
        if 'aes_args' in locals() and aes_args and 'code' in locals():
            export_code = code
            
            st.markdown("**Complete Code:**")
            st.code(export_code, language="python")
            
            st.download_button(
                "Download Code",
                export_code,
                file_name="my_ggplot.py",
                mime="text/plain"
            )
            
            st.markdown("---")
            st.markdown("**Quick Copy Snippets:**")
            
            aes_params = ', '.join([f'{k}="{v}"' for k, v in aes_args.items()])
            plot_only = f"ggplot(df, aes({aes_params})) + {geom_display}(alpha={alpha_val})"
            st.code(plot_only, language="python")
            
            st.caption("Tip: Copy the full code above to get all layers including themes and advanced features!")
        else:
            st.info("Build a plot above to generate exportable code!")

def main():
    st.markdown("""
    <style>
    /* Clean, professional background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content area - clean white cards */
    .main .block-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Header styling - clean dark text */
    h1 {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Clean sidebar - comprehensive styling */
    .css-1d391kg, [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar text styling - all possible selectors with maximum specificity */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        color: #1e293b !important;
    }
    
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span, .css-1d391kg label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #374151 !important;
    }
    
    /* Sidebar section headers */
    .css-1d391kg .css-10trblm, [data-testid="stSidebar"] .css-10trblm, section[data-testid="stSidebar"] .css-10trblm {
        color: #1e293b !important;
    }
    
    /* Force all sidebar text to be visible with maximum specificity */
    .css-1d391kg *, [data-testid="stSidebar"] *, section[data-testid="stSidebar"] * {
        color: #374151 !important;
    }
    
    /* Override any inherited white text */
    .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stMarkdown {
        color: #374151 !important;
    }
    
    /* Specific fixes for common Streamlit sidebar elements */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #374151 !important;
    }
    
    /* Sidebar widget text */
    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stRadio div,
    section[data-testid="stSidebar"] .stCheckbox div {
        color: #374151 !important;
    }
    
    /* Sidebar markdown text */
    section[data-testid="stSidebar"] .css-1v0mbdj, 
    section[data-testid="stSidebar"] .css-16idsys {
        color: #374151 !important;
    }
    
    /* Modern tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8fafc;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: #64748b !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #2563eb !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Modern button styling */
    .stButton > button {
        background: #2563eb;
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Clean form inputs */
    .stSelectbox > div > div {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        color: #374151;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    .stTextInput > div > div > input {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        color: #374151;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Clean expanders */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* Card-style metrics */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Clean data frames */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        border-radius: 4px;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background: white;
    }
    
    /* Success alerts */
    .stAlert[data-baseweb="notification"] div[data-baseweb="notification"] {
        background: #f0fdf4;
        border-color: #bbf7d0;
    }
    
    /* Info alerts */
    .stAlert[data-baseweb="notification"]:has([data-testid="stNotificationContentInfo"]) {
        background: #eff6ff;
        border-color: #bfdbfe;
    }
    
    /* Warning alerts */
    .stAlert[data-baseweb="notification"]:has([data-testid="stNotificationContentWarning"]) {
        background: #fffbeb;
        border-color: #fed7aa;
    }
    
    /* Remove unnecessary margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Additional sidebar fixes for maximum compatibility */
    section[data-testid="stSidebar"] {
        background-color: white !important;
    }
    
    section[data-testid="stSidebar"] .css-1v0mbdj p,
    section[data-testid="stSidebar"] .css-16idsys p,
    section[data-testid="stSidebar"] .element-container,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #374151 !important;
    }
    
    /* Force visibility for all sidebar content */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown *,
    section[data-testid="stSidebar"] .element-container *,
    section[data-testid="stSidebar"] .css-1v0mbdj,
    section[data-testid="stSidebar"] .css-16idsys {
        color: #374151 !important;
        background: transparent !important;
    }
    
    /* Ensure sidebar headers are visible */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label > div {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 4px;
    }
    
    /* Radio button styling */
    .stRadio > label > div {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Professional Data Analysis Suite")
    st.markdown("**Streamlined data analysis and visualization platform for researchers and analysts**")
    
    st.sidebar.write("## Performance Settings")
    enable_caching = st.sidebar.checkbox("Enable visualization caching", value=True)
    lazy_loading = st.sidebar.checkbox("Lazy load visualizations", value=True)
    use_sampling = st.sidebar.checkbox("Use data sampling for large datasets", value=True)
    
    with st.sidebar:
        st.header("Data Upload")
        
        st.subheader("Sample Data")
        sample_data = st.selectbox(
            "Try sample datasets:",
            ["None", "Employee Data", "Sales Data"],
            help="Load pre-built sample datasets to explore the app's features"
        )
        
        if sample_data != "None":
            try:
                with st.spinner("Loading sample data..."):
                    if sample_data == "Employee Data":
                        df = pd.read_csv("data/sample_employee_data.csv")
                        filename = "sample_employee_data.csv"
                    elif sample_data == "Sales Data":
                        df = pd.read_csv("data/sample_sales_data.csv")
                        filename = "sample_sales_data.csv"
                    
                    df = process_dataframe(df)
                    st.session_state['df'] = df
                    st.session_state['filename'] = filename
                    st.session_state['enable_caching'] = enable_caching
                    
                    if enable_caching:
                        with st.spinner("Precomputing analyses for faster visualization..."):
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            if len(numeric_cols) > 1:
                                st.session_state['cached_corr'] = cached_correlation_matrix(df, numeric_cols)
                            if len(numeric_cols) > 0:
                                st.session_state['cached_desc'] = cached_describe(df, numeric_cols)
                    
                    st.success(f"Loaded {filename}")
                
            except FileNotFoundError:
                st.error(f"Sample file not found. Please upload your own data.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            with st.spinner("Processing data..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            df = pd.read_csv(uploaded_file, low_memory=True)
                        except:
                            df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file, sheet_name=0)
                    
                    original_row_count = len(df)
                    
                    if use_sampling and len(df) > 10000:
                        sample_size = st.sidebar.slider("Sample size (rows)", 
                                                   min_value=1000, 
                                                   max_value=min(len(df), 100000), 
                                                   value=min(10000, len(df)),
                                                   step=1000)
                        df = df.sample(sample_size, random_state=42)
                        st.sidebar.info(f"Sampled {sample_size} rows from {original_row_count} total rows")
                    
                    df = process_dataframe(df)
                    st.session_state['df'] = df
                    st.session_state['filename'] = uploaded_file.name
                    st.session_state['enable_caching'] = enable_caching
                    
                    if enable_caching:
                        with st.spinner("Precomputing analyses for faster visualization..."):
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            if len(numeric_cols) > 1:
                                st.session_state['cached_corr'] = cached_correlation_matrix(df, numeric_cols)
                            if len(numeric_cols) > 0:
                                st.session_state['cached_desc'] = cached_describe(df, numeric_cols)
                    
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

            tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Visualization", "Statistical Analysis", "ggplot Playground"])
            
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
                        ["Box Plot", "Cluster Visualization", "Correlation Analysis", "Distribution Analysis", 
                         "Heatmap", "Parallel Coordinates", "Scatter Plot", 
                         "3D Scatter Plot", "Time Series Analysis", "Violin Plot"]
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
                        
                        st.write("## Configuration")
                        
                        plot_type = st.radio("Plot type:", ["Single Column", "Compare Multiple Columns"], horizontal=True)
                        
                        if plot_type == "Single Column":
                            y_col = st.selectbox("Select column for analysis:", numeric_cols)
                            
                            cat_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            if cat_cols:
                                x_col = st.selectbox("Group by category (optional):", ["None"] + cat_cols)
                                
                                with st.expander("Advanced Options"):
                                    show_points = st.checkbox("Show all data points", value=True)
                                    point_size = st.slider("Point size:", 1, 10, 4) if show_points else 4
                                    show_mean = st.checkbox("Show mean marker", value=True)
                                    use_color = st.checkbox("Use custom color", value=False)
                                    if use_color:
                                        box_color = st.color_picker("Box color:", "#636EFA")
                                    else:
                                        box_color = "#636EFA"
                                
                                if x_col != "None":
                                    value_counts = df_selected[x_col].value_counts()
                                    
                                    if len(value_counts) > 10:
                                        limit_cats = st.checkbox(f"Limit to top categories (found {len(value_counts)} categories)", value=True)
                                        if limit_cats:
                                            top_n = st.slider("Number of top categories to show:", 2, 20, 10)
                                            top_cats = value_counts.nlargest(top_n).index.tolist()
                                            filtered_df = df_selected[df_selected[x_col].isin(top_cats)]
                                            st.info(f"Showing top {top_n} categories out of {len(value_counts)}")
                                        else:
                                            filtered_df = df_selected
                                    else:
                                        filtered_df = df_selected
                                    
                                    fig = px.box(
                                        filtered_df, 
                                        x=x_col, 
                                        y=y_col, 
                                        color=x_col if not use_color else None,
                                        template="plotly_dark",
                                        points="all" if show_points else "outliers",
                                        title=f"Box Plot: {y_col} by {x_col}",
                                        notched=False
                                    )
                                    
                                    fig.update_traces(
                                        boxpoints="all" if show_points else "outliers",
                                        jitter=0.3,
                                        pointpos=-1.8,
                                        marker=dict(size=point_size, opacity=0.6)
                                    )
                                    
                                    if use_color:
                                        for i in range(len(fig.data)):
                                            fig.data[i].marker.color = box_color
                                            fig.data[i].line.color = box_color
                                    
                                    if show_mean:
                                        means = filtered_df.groupby(x_col)[y_col].mean().reset_index()
                                        fig.add_trace(
                                            go.Scatter(
                                                x=means[x_col],
                                                y=means[y_col],
                                                mode='markers',
                                                marker=dict(
                                                    symbol='diamond',
                                                    size=10,
                                                    color='white',
                                                    line=dict(width=2, color='red')
                                                ),
                                                name='Mean'
                                            )
                                        )
                                else:
                                    fig = px.box(
                                        df_selected, 
                                        y=y_col, 
                                        template="plotly_dark",
                                        points="all" if show_points else "outliers",
                                        title=f"Box Plot: {y_col}",
                                        notched=False
                                    )
                                    
                                    fig.update_traces(
                                        boxpoints="all" if show_points else "outliers",
                                        jitter=0.3,
                                        pointpos=-1.8,
                                        marker=dict(size=point_size, opacity=0.6)
                                    )
                                    
                                    if use_color:
                                        fig.update_traces(marker_color=box_color, line_color=box_color)
                                    
                                    if show_mean:
                                        mean_val = df_selected[y_col].mean()
                                        fig.add_shape(
                                            type="line",
                                            x0=-0.5, x1=0.5,
                                            y0=mean_val, y1=mean_val,
                                            line=dict(color="red", width=3, dash="dash"),
                                        )
                                        
                                        fig.add_annotation(
                                            x=0.5, y=mean_val,
                                            text=f"Mean: {mean_val:.2f}",
                                            showarrow=False,
                                            xshift=50,
                                            font=dict(color="white")
                                        )
                            else:
                                with st.expander("Advanced Options"):
                                    show_points = st.checkbox("Show all data points", value=True)
                                    point_size = st.slider("Point size:", 1, 10, 4) if show_points else 4
                                    show_mean = st.checkbox("Show mean marker", value=True)
                                    use_color = st.checkbox("Use custom color", value=False)
                                    if use_color:
                                        box_color = st.color_picker("Box color:", "#636EFA")
                                    else:
                                        box_color = "#636EFA"
                                        
                                fig = px.box(
                                    df_selected, 
                                    y=y_col, 
                                    template="plotly_dark",
                                    points="all" if show_points else "outliers",
                                    title=f"Box Plot: {y_col}",
                                    notched=False
                                )
                                
                                fig.update_traces(
                                    boxpoints="all" if show_points else "outliers",
                                    jitter=0.3,
                                    pointpos=-1.8,
                                    marker=dict(size=point_size, opacity=0.6)
                                )
                                
                                if use_color:
                                    fig.update_traces(marker_color=box_color, line_color=box_color)
                                
                                if show_mean:
                                    mean_val = df_selected[y_col].mean()
                                    fig.add_shape(
                                        type="line",
                                        x0=-0.5, x1=0.5,
                                        y0=mean_val, y1=mean_val,
                                        line=dict(color="red", width=3, dash="dash"),
                                    )
                                    
                                    fig.add_annotation(
                                        x=0.5, y=mean_val,
                                        text=f"Mean: {mean_val:.2f}",
                                        showarrow=False,
                                        xshift=50,
                                        font=dict(color="white")
                                    )
                        else:
                            selected_cols = st.multiselect(
                                "Select columns to compare:", 
                                numeric_cols,
                                default=numeric_cols[:min(5, len(numeric_cols))]
                            )
                            
                            if not selected_cols:
                                st.warning("Please select at least one column for comparison.")
                            else:
                                with st.expander("Advanced Options"):
                                    show_points = st.checkbox("Show all data points", value=True)
                                    point_size = st.slider("Point size:", 1, 10, 4) if show_points else 4
                                    show_mean = st.checkbox("Show mean markers", value=True)
                                    use_color_scale = st.checkbox("Use color scale", value=True)
                                    orient = st.radio("Orientation:", ["Vertical", "Horizontal"], horizontal=True)
                                    
                                plot_data = pd.melt(
                                    df_selected[selected_cols], 
                                    var_name='Column', 
                                    value_name='Value'
                                )
                                
                                if orient == "Vertical":
                                    fig = px.box(
                                        plot_data, 
                                        x='Column', 
                                        y='Value',
                                        color='Column' if use_color_scale else None,
                                        template="plotly_dark",
                                        points="all" if show_points else "outliers",
                                        title="Box Plot Comparison",
                                        notched=False
                                    )
                                    
                                    fig.update_traces(
                                        boxpoints="all" if show_points else "outliers",
                                        jitter=0.3,
                                        pointpos=-1.8,
                                        marker=dict(size=point_size, opacity=0.6)
                                    )
                                else:
                                    fig = px.box(
                                        plot_data, 
                                        y='Column', 
                                        x='Value',
                                        color='Column' if use_color_scale else None, 
                                        template="plotly_dark",
                                        points="all" if show_points else "outliers",
                                        title="Box Plot Comparison",
                                        notched=False
                                    )
                                    
                                    fig.update_traces(
                                        boxpoints="all" if show_points else "outliers",
                                        jitter=0.3,
                                        pointpos=-1.8,
                                        marker=dict(size=point_size, opacity=0.6)
                                    )
                                
                                if show_mean:
                                    means = plot_data.groupby('Column')['Value'].mean().reset_index()
                                    if orient == "Vertical":
                                        fig.add_trace(
                                            go.Scatter(
                                                x=means['Column'],
                                                y=means['Value'],
                                                mode='markers',
                                                marker=dict(
                                                    symbol='diamond',
                                                    size=10,
                                                    color='white',
                                                    line=dict(width=2, color='red')
                                                ),
                                                name='Mean'
                                            )
                                        )
                                    else:
                                        fig.add_trace(
                                            go.Scatter(
                                                y=means['Column'],
                                                x=means['Value'],
                                                mode='markers',
                                                marker=dict(
                                                    symbol='diamond',
                                                    size=10,
                                                    color='white',
                                                    line=dict(width=2, color='red')
                                                ),
                                                name='Mean'
                                            )
                                        )
                        
                        fig.update_layout(
                            height=600,
                            plot_bgcolor='rgba(17, 17, 17, 0.8)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=12,
                                font_family="Rockwell"
                            ),
                            boxgap=0.3,
                            boxgroupgap=0.2
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128, 128, 128, 0.2)')
                        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128, 128, 128, 0.2)')
                        
                        with st.expander("Visual Options"):
                            show_whiskers = st.checkbox("Show whiskers", value=True)
                            if not show_whiskers:
                                for i in range(len(fig.data)):
                                    if hasattr(fig.data[i], 'line'):
                                        fig.data[i].line.width = 0
                                    if hasattr(fig.data[i], 'whiskerwidth'):
                                        fig.data[i].whiskerwidth = 0
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("## Statistical Summary")
                        
                        if plot_type == "Single Column":
                            stats = df_selected[y_col].describe()
                            iqr = stats['75%'] - stats['25%']
                            stats_df = pd.DataFrame({
                                'Metric': ['Count', 'Mean', 'Standard Deviation', 'Minimum', '25%', 'Median (50%)', '75%', 'Maximum', 'IQR'],
                                'Value': [
                                    stats['count'], 
                                    stats['mean'], 
                                    stats['std'], 
                                    stats['min'], 
                                    stats['25%'], 
                                    stats['50%'], 
                                    stats['75%'], 
                                    stats['max'],
                                    iqr
                                ]
                            })
                            
                            lower_bound = stats['25%'] - 1.5 * iqr
                            upper_bound = stats['75%'] + 1.5 * iqr
                            outliers = df_selected[(df_selected[y_col] < lower_bound) | (df_selected[y_col] > upper_bound)][y_col]
                            
                            st.dataframe(stats_df)
                            
                            if len(outliers) > 0:
                                st.write(f"**Found {len(outliers)} outliers** (values outside 1.5 × IQR)")
                                st.write(f"Outlier threshold: < {lower_bound:.2f} or > {upper_bound:.2f}")
                                
                                show_outliers = st.checkbox("Show outlier details")
                                if show_outliers:
                                    st.dataframe(outliers.reset_index())
                        else:
                            stats_dict = {}
                            for col in selected_cols:
                                col_stats = df_selected[col].describe()
                                iqr = col_stats['75%'] - col_stats['25%']
                                
                                stats_dict[col] = {
                                    'Count': col_stats['count'],
                                    'Mean': col_stats['mean'],
                                    'Standard Deviation': col_stats['std'],
                                    'Minimum': col_stats['min'],
                                    '25%': col_stats['25%'],
                                    'Median (50%)': col_stats['50%'],
                                    '75%': col_stats['75%'],
                                    'Maximum': col_stats['max'],
                                    'IQR': iqr,
                                    'Outliers': len(df_selected[(df_selected[col] < col_stats['25%'] - 1.5 * iqr) | 
                                                        (df_selected[col] > col_stats['75%'] + 1.5 * iqr)])
                                }
                            
                            stats_df = pd.DataFrame(stats_dict)
                            st.dataframe(stats_df)
                    
                    elif viz_type == "3D Scatter Plot":
                        plot_3d_scatter(df_selected, numeric_cols)
                    
                    elif viz_type == "Heatmap":
                        plot_heatmap(df_selected, numeric_cols)
                    
                    elif viz_type == "Parallel Coordinates":
                        plot_parallel_coordinates(df_selected, numeric_cols)
                        
                    elif viz_type == "Violin Plot":
                        plot_violin(df_selected, numeric_cols)
                        
                    elif viz_type == "Cluster Visualization":
                        plot_cluster_visualization(df_selected, numeric_cols)
            
            with tab3:
                st.header("Statistical Analysis")
                
                numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if len(numeric_cols) > 0:
                    st.subheader("Descriptive Statistics")
                    if enable_caching and 'cached_desc' in st.session_state and set(numeric_cols).issubset(st.session_state['cached_desc'].columns[1:]):
                        desc_stats = st.session_state['cached_desc'].reset_index()
                    else:
                        desc_stats = cached_describe(df_selected, numeric_cols).reset_index()
                    
                    desc_stats.columns = ['Statistic'] + numeric_cols
                    st.dataframe(desc_stats)
                    
                    if len(numeric_cols) >= 2:
                        st.subheader("Correlation Analysis")
                        if enable_caching and 'cached_corr' in st.session_state and set(numeric_cols).issubset(st.session_state['cached_corr'].columns):
                            corr = st.session_state['cached_corr']
                        else:
                            corr = cached_correlation_matrix(df_selected, numeric_cols)
                        
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
            
            with tab4:
                st.header("ggplot Playground")
                if PLOTNINE_AVAILABLE:
                    ggplot_playground(df_selected)
                else:
                    st.warning("plotnine is not available. Please install it to use the ggplot Playground.")
                    st.code("pip install plotnine==0.14.5", language="bash")
                    st.info("The ggplot Playground allows you to interactively build beautiful visualizations using the Grammar of Graphics approach.")
    else:
        st.info("Please upload a file from the sidebar to begin.")
        st.markdown("")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or uploading a different file.")