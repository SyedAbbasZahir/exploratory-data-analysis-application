"""
Streamlit EDA Application
A comprehensive tool for Exploratory Data Analysis with interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced EDA Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3498db;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

# Caching functions for performance
@st.cache_data
def load_data(uploaded_file):
    """Load and cache the uploaded dataset."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def compute_correlation(df, method='pearson'):
    """Compute and cache correlation matrix."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        return df[numeric_cols].corr(method=method)
    return None

@st.cache_data
def generate_pairplot_data(df, columns):
    """Prepare data for pair plot (sample if too large)."""
    if len(df) > 1000:
        return df[columns].sample(1000, random_state=42)
    return df[columns]

def get_column_types(df):
    """Categorize columns into numeric and categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols

def generate_insights(df):
    """Automatically generate data insights."""
    insights = []
    
    # Missing values insight
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    if missing.any():
        max_missing_col = missing_pct.idxmax()
        max_missing_val = missing_pct.max()
        insights.append(f"⚠️ **Missing Data Alert**: '{max_missing_col}' has the highest missing value rate at {max_missing_val}%")
    
    # Correlation insight
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.unstack().sort_values(ascending=False).iloc[0]
        max_corr_cols = corr_matrix.unstack().sort_values(ascending=False).index[0]
        if max_corr > 0.7:
            insights.append(f"🔗 **Strong Correlation**: '{max_corr_cols[0]}' and '{max_corr_cols[1]}' are highly correlated (r={max_corr:.2f})")
    
    # Duplicates insight
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        insights.append(f"🔄 **Duplicates Found**: {duplicates} duplicate rows detected ({duplicates/len(df)*100:.1f}%)")
    
    # Data type insights
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                insights.append(f"🆔 **Potential ID Column**: '{col}' has {df[col].nunique()} unique values ({unique_ratio*100:.0f}%)")
    
    return insights

# Sidebar Navigation
st.sidebar.markdown("## 📊 Navigation")
sections = [
    "🏠 Home",
    "📁 Data Upload", 
    "📋 Dataset Overview",
    "🧹 Data Cleaning",
    "📈 Visualizations",
    "🔍 Insights"
]
selected_section = st.sidebar.radio("Go to", sections)

# Main Content Area
st.markdown('<div class="main-header">🔬 Advanced EDA Tool</div>', unsafe_allow_html=True)
st.markdown("Upload your dataset to perform comprehensive exploratory data analysis with interactive visualizations.")

# Section 1: Home
if selected_section == "🏠 Home":
    st.markdown('<div class="section-header">Welcome to the Advanced EDA Tool</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Data Upload</h3>
            <p>Upload CSV files and preview your data instantly with automatic type detection.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧹 Smart Cleaning</h3>
            <p>Handle missing values, remove duplicates, and filter columns with an intuitive interface.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Interactive Viz</h3>
            <p>Create publication-ready plots with Plotly. Export and customize every visualization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👈 Use the sidebar to navigate through different sections. Start by uploading your dataset!")

# Section 2: Data Upload
elif selected_section == "📁 Data Upload":
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Upload your dataset in CSV format")
    
    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            df = load_data(uploaded_file)
            
        if df is not None:
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.df_cleaned = df.copy()
            
            st.success(f"✅ Successfully loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")
            
            # Dataset Preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic Info
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Column Information
            st.subheader("Column Schema")
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(schema_df, use_container_width=True)
    else:
        st.info("📤 Please upload a CSV file to begin analysis")

# Section 3: Dataset Overview
elif selected_section == "📋 Dataset Overview":
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df_cleaned
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔍 Missing Values", "📝 Data Sample"])
        
        with tab1:
            st.subheader("Statistical Summary")
            
            # Numeric statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("**Numeric Columns**")
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
            
            # Categorical statistics
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                st.markdown("**Categorical Columns**")
                cat_stats = pd.DataFrame({
                    'Column': categorical_cols,
                    'Unique Values': [df[col].nunique() for col in categorical_cols],
                    'Most Frequent': [df[col].mode()[0] if not df[col].mode().empty else 'N/A' for col in categorical_cols],
                    'Frequency': [df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0 for col in categorical_cols]
                })
                st.dataframe(cat_stats, use_container_width=True)
        
        with tab2:
            st.subheader("Missing Values Analysis")
            
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing Count']
            missing_data['Missing %'] = (missing_data['Missing Count'] / len(df) * 100).round(2)
            missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            if not missing_data.empty:
                st.dataframe(missing_data, use_container_width=True)
                
                # Visualization
                fig = px.bar(missing_data, x='Column', y='Missing %', 
                            title='Missing Values by Column (%)',
                            color='Missing %', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        with tab3:
            st.subheader("Data Sample")
            sample_size = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(df.head(sample_size), use_container_width=True)

# Section 4: Data Cleaning
elif selected_section == "🧹 Data Cleaning":
    st.markdown('<div class="section-header">🧹 Data Cleaning Tools</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df_cleaned.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Handle Missing Values")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                selected_col = st.selectbox("Select column with missing values", missing_cols)
                
                strategy = st.radio("Cleaning strategy", 
                                  ["Drop rows", "Fill with Mean", "Fill with Median", 
                                   "Fill with Mode", "Fill with Custom Value"])
                
                if st.button("Apply Missing Value Treatment"):
                    with st.spinner("Applying changes..."):
                        if strategy == "Drop rows":
                            df = df.dropna(subset=[selected_col])
                        elif strategy == "Fill with Mean":
                            df[selected_col] = df[selected_col].fillna(df[selected_col].mean())
                        elif strategy == "Fill with Median":
                            df[selected_col] = df[selected_col].fillna(df[selected_col].median())
                        elif strategy == "Fill with Mode":
                            mode_val = df[selected_col].mode()
                            if not mode_val.empty:
                                df[selected_col] = df[selected_col].fillna(mode_val[0])
                        elif strategy == "Fill with Custom Value":
                            custom_val = st.text_input("Enter custom value")
                            if custom_val:
                                df[selected_col] = df[selected_col].fillna(custom_val)
                        
                        st.session_state.df_cleaned = df
                        st.success(f"✅ Applied {strategy} to '{selected_col}'")
                        st.rerun()
            else:
                st.info("No missing values to handle!")
            
            # Remove duplicates
            st.subheader("Remove Duplicates")
            duplicates = df.duplicated().sum()
            st.write(f"Duplicate rows found: **{duplicates}**")
            
            if duplicates > 0 and st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.session_state.df_cleaned = df
                st.success("✅ Duplicates removed!")
                st.rerun()
        
        with col2:
            st.subheader("Column Operations")
            
            # Column filtering
            st.markdown("**Select columns to keep**")
            all_cols = df.columns.tolist()
            selected_cols = st.multiselect("Columns", all_cols, default=all_cols)
            
            if st.button("Filter Columns"):
                if selected_cols:
                    df = df[selected_cols]
                    st.session_state.df_cleaned = df
                    st.success(f"✅ Kept {len(selected_cols)} columns")
                    st.rerun()
            
            # Drop specific column
            st.markdown("**Drop specific column**")
            col_to_drop = st.selectbox("Select column to drop", df.columns, key='drop_col')
            if st.button("Drop Column"):
                df = df.drop(columns=[col_to_drop])
                st.session_state.df_cleaned = df
                st.success(f"✅ Dropped '{col_to_drop}'")
                st.rerun()
        
        # Reset button
        if st.button("🔄 Reset to Original Data", type="secondary"):
            st.session_state.df_cleaned = st.session_state.original_df.copy()
            st.success("✅ Data reset to original state!")
            st.rerun()

# Section 5: Visualizations
elif selected_section == "📈 Visualizations":
    st.markdown('<div class="section-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df_cleaned
        numeric_cols, categorical_cols, _ = get_column_types(df)
        
        viz_type = st.sidebar.selectbox("Select Visualization Type", 
                                       ["Distribution", "Box Plot", "Correlation Heatmap", 
                                        "Scatter Plot", "Bar Chart", "Pair Plot"])
        
        if viz_type == "Distribution":
            st.subheader("Distribution Plot")
            
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(df, x=selected_col, nbins=30, 
                                     marginal="box", 
                                     title=f"Distribution of {selected_col}",
                                     opacity=0.7)
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # KDE plot using plotly
                    fig = ff.create_distplot([df[selected_col].dropna()], [selected_col], 
                                           show_hist=False, show_rug=False)
                    fig.update_layout(title=f"Density Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                stats = df[selected_col].describe()
                st.write("**Statistics:**")
                st.write(f"Mean: {stats['mean']:.2f} | Median: {df[selected_col].median():.2f} | Std: {stats['std']:.2f}")
            else:
                st.warning("No numeric columns available for distribution plot")
        
        elif viz_type == "Box Plot":
            st.subheader("Box Plot")
            
            if numeric_cols:
                y_col = st.selectbox("Select value column (Y-axis)", numeric_cols)
                
                if categorical_cols:
                    x_col = st.selectbox("Select category column (X-axis)", ["None"] + categorical_cols)
                else:
                    x_col = "None"
                
                if x_col != "None":
                    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                else:
                    fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available")
        
        elif viz_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            
            if len(numeric_cols) > 1:
                method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
                
                with st.spinner("Computing correlation matrix..."):
                    corr_matrix = compute_correlation(df, method)
                
                # Plotly heatmap
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title=f"Correlation Matrix ({method.capitalize()})",
                              color_continuous_scale='RdBu_r',
                              zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.subheader("Top Correlations")
                corr_pairs = corr_matrix.unstack().sort_values(ascending=False).reset_index()
                corr_pairs = corr_pairs[corr_pairs['level_0'] != corr_pairs['level_1']]
                corr_pairs['pair'] = corr_pairs['level_0'] + ' - ' + corr_pairs['level_1']
                corr_pairs = corr_pairs.drop_duplicates(subset=[0])
                st.dataframe(corr_pairs.head(10)[['pair', 0]].rename(columns={0: 'Correlation'}), 
                           use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        elif viz_type == "Scatter Plot":
            st.subheader("Scatter Plot")
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", numeric_cols, key='scatter_x')
                y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key='scatter_y')
                
                color_col = None
                if categorical_cols:
                    color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    if color_col == "None":
                        color_col = None
                
                size_col = None
                size_options = ["None"] + [c for c in numeric_cols if c not in [x_col, y_col]]
                size_selection = st.selectbox("Size by (optional)", size_options)
                if size_selection != "None":
                    size_col = size_selection
                
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                               title=f"{y_col} vs {x_col}", opacity=0.6)
                fig.update_traces(marker=dict(size=8 if size_col is None else None))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation
                corr_val = df[[x_col, y_col]].corr().iloc[0, 1]
                st.info(f"**Correlation coefficient**: {corr_val:.3f}")
            else:
                st.warning("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "Bar Chart":
            st.subheader("Bar Chart")
            
            if categorical_cols:
                x_col = st.selectbox("Category column", categorical_cols)
                
                agg_options = ["Count", "Mean", "Sum", "Median"]
                agg_func = st.selectbox("Aggregation", agg_options)
                
                if agg_func == "Count":
                    value_counts = df[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'Count']
                    fig = px.bar(value_counts, x=x_col, y='Count', 
                               title=f"Count of {x_col}",
                               color='Count', color_continuous_scale='Viridis')
                else:
                    if numeric_cols:
                        y_col = st.selectbox("Value column", numeric_cols)
                        if agg_func == "Mean":
                            grouped = df.groupby(x_col)[y_col].mean().reset_index()
                        elif agg_func == "Sum":
                            grouped = df.groupby(x_col)[y_col].sum().reset_index()
                        elif agg_func == "Median":
                            grouped = df.groupby(x_col)[y_col].median().reset_index()
                        
                        fig = px.bar(grouped, x=x_col, y=y_col, 
                                   title=f"{agg_func} of {y_col} by {x_col}",
                                   color=y_col, color_continuous_scale='Viridis')
                    else:
                        st.error("Need numeric columns for aggregation")
                        fig = None
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns available")
        
        elif viz_type == "Pair Plot":
            st.subheader("Pair Plot")
            
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Select columns for pair plot (2-5 recommended)", 
                                             numeric_cols, 
                                             default=numeric_cols[:3])
                
                if len(selected_cols) >= 2:
                    if len(selected_cols) > 5:
                        st.warning("⚠️ Selecting more than 5 columns may cause performance issues")
                    
                    with st.spinner("Generating pair plot..."):
                        pair_data = generate_pairplot_data(df, selected_cols)
                        
                        # Use Plotly scatter matrix
                        fig = px.scatter_matrix(pair_data, 
                                              dimensions=selected_cols,
                                              title="Pair Plot Matrix",
                                              opacity=0.6)
                        fig.update_traces(diagonal_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select at least 2 columns")
            else:
                st.warning("Need at least 2 numeric columns")

# Section 6: Insights
elif selected_section == "🔍 Insights":
    st.markdown('<div class="section-header">🔍 Automated Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df_cleaned
        
        with st.spinner("Generating insights..."):
            insights = generate_insights(df)
        
        if insights:
            st.subheader("Key Findings")
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.success("✅ No major issues detected in the dataset!")
        
        # Additional Analysis
        st.subheader("Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Columns Summary**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].agg(['min', 'max', 'mean', 'std']).T
                st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.markdown("**High Cardinality Columns**")
            high_card = []
            for col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5:
                    high_card.append({
                        'Column': col,
                        'Unique Values': df[col].nunique(),
                        'Unique %': f"{unique_ratio*100:.1f}%"
                    })
            if high_card:
                st.dataframe(pd.DataFrame(high_card), use_container_width=True)
            else:
                st.info("No high cardinality columns found")
        
        # Data Quality Score
        st.subheader("Data Quality Score")
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        completeness = (1 - missing_cells / total_cells) * 100
        uniqueness = (1 - duplicate_rows / len(df)) * 100 if len(df) > 0 else 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Completeness", f"{completeness:.1f}%")
        col2.metric("Uniqueness", f"{uniqueness:.1f}%")
        col3.metric("Overall Score", f"{(completeness + uniqueness) / 2:.1f}%")
        
        # Progress bars
        st.progress(completeness / 100, text=f"Completeness: {completeness:.1f}%")
        st.progress(uniqueness / 100, text=f"Uniqueness: {uniqueness:.1f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This EDA tool is built with Streamlit and Plotly. "
    "Upload any CSV file to explore your data interactively."
)