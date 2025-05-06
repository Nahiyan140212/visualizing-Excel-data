import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Excel Data Visualizer",
    page_icon="📊",
    layout="wide"
)

# Add a title and description
st.title("📊 Excel Data Visualizer")
st.markdown("""
Upload your Excel file and create interactive visualizations instantly!
This app supports various chart types and data analysis features.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load the data
    try:
        df = pd.read_excel(uploaded_file)
        
        # Display basic info about the dataset
        st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📊 Visualizations", "📈 Time Series", "🔍 Data Analysis"])
        
        with tab1:
            st.header("Data Preview")
            # Show data info and preview
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
                dtypes_df.index.name = "Column"
                st.dataframe(dtypes_df.reset_index())
                
            with col2:
                st.subheader("Summary Statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe())
                else:
                    st.info("No numeric columns found for summary statistics.")
            
            # Display raw data with pagination
            st.subheader("Raw Data")
            page_size = st.slider("Rows per page", 5, 50, 10)
            page_number = st.number_input("Page", min_value=1, max_value=max(1, (len(df) // page_size) + 1), step=1)
            
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
        
        with tab2:
            st.header("Create Visualizations")
            
            chart_type = st.selectbox(
                "Select chart type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Heatmap", "3D Scatter"]
            )
            
            # Get column types for better suggestions
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # For datetime columns that might be stored as objects
            for col in df.columns:
                if col not in datetime_cols:
                    try:
                        if pd.to_datetime(df[col], errors='coerce').notna().all():
                            datetime_cols.append(col)
                    except:
                        pass
            
            # Create two columns for chart controls
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                    x_axis = st.selectbox("X-axis", df.columns)
                    y_axis = st.selectbox("Y-axis", numeric_cols if numeric_cols else df.columns)
                    
                    color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    color_col = None if color_option == "None" else color_option
                    
                    # Additional options for specific chart types
                    if chart_type == "Bar Chart":
                        orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
                    
                    if chart_type == "Scatter Plot":
                        size_option = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
                        size_col = None if size_option == "None" else size_option
                
                elif chart_type == "Histogram":
                    column = st.selectbox("Column", numeric_cols if numeric_cols else df.columns)
                    bins = st.slider("Number of bins", 5, 100, 20)
                    
                elif chart_type == "Box Plot":
                    y_axis = st.selectbox("Values", numeric_cols if numeric_cols else df.columns)
                    x_axis = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                    x_axis = None if x_axis == "None" else x_axis
                
                elif chart_type == "Pie Chart":
                    values = st.selectbox("Values", numeric_cols if numeric_cols else df.columns)
                    names = st.selectbox("Names", categorical_cols if categorical_cols else df.columns)
                    
                elif chart_type == "Heatmap":
                    corr_method = st.radio("Correlation Method", ["pearson", "kendall", "spearman"])
                    
                elif chart_type == "3D Scatter":
                    x_axis = st.selectbox("X-axis", numeric_cols if numeric_cols else df.columns)
                    y_axis = st.selectbox("Y-axis", numeric_cols if numeric_cols else df.columns)
                    z_axis = st.selectbox("Z-axis", numeric_cols if numeric_cols else df.columns)
                    color_option = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    color_col = None if color_option == "None" else color_option
            
            with col2:
                # Generate the selected chart
                st.subheader(f"{chart_type} Visualization")
                
                try:
                    if chart_type == "Bar Chart":
                        if orientation == "Vertical":
                            if color_col:
                                fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
                            else:
                                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                        else:  # Horizontal
                            if color_col:
                                fig = px.bar(df, x=y_axis, y=x_axis, color=color_col, orientation='h', title=f"{y_axis} by {x_axis}")
                            else:
                                fig = px.bar(df, x=y_axis, y=x_axis, orientation='h', title=f"{y_axis} by {x_axis}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Line Chart":
                        if color_col:
                            fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis}")
                        else:
                            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Scatter Plot":
                        if color_col and size_col:
                            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, size=size_col, 
                                            title=f"{y_axis} vs {x_axis}", 
                                            hover_data=df.columns)
                        elif color_col:
                            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, 
                                            title=f"{y_axis} vs {x_axis}", 
                                            hover_data=df.columns)
                        elif size_col:
                            fig = px.scatter(df, x=x_axis, y=y_axis, size=size_col, 
                                            title=f"{y_axis} vs {x_axis}", 
                                            hover_data=df.columns)
                        else:
                            fig = px.scatter(df, x=x_axis, y=y_axis, 
                                            title=f"{y_axis} vs {x_axis}", 
                                            hover_data=df.columns)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Box Plot":
                        if x_axis:
                            fig = px.box(df, x=x_axis, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}")
                        else:
                            fig = px.box(df, y=y_axis, title=f"Distribution of {y_axis}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Pie Chart":
                        # Aggregate data if needed
                        if df[names].nunique() > 15:
                            st.warning("Too many categories for a pie chart. Showing top 15 by value.")
                            pie_data = df.groupby(names)[values].sum().nlargest(15).reset_index()
                        else:
                            pie_data = df.groupby(names)[values].sum().reset_index()
                        
                        fig = px.pie(pie_data, values=values, names=names, title=f"{values} by {names}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Heatmap":
                        # Only use numeric columns for correlation
                        if len(numeric_cols) < 2:
                            st.error("Not enough numeric columns for a correlation heatmap.")
                        else:
                            corr = df[numeric_cols].corr(method=corr_method)
                            fig = px.imshow(corr, text_auto=True, 
                                          title=f"Correlation Heatmap ({corr_method})",
                                          color_continuous_scale='RdBu_r')
                            st.plotly_chart(fig, use_container_width=True)
                            
                    elif chart_type == "3D Scatter":
                        if color_col:
                            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, 
                                              color=color_col, title=f"3D Scatter Plot")
                        else:
                            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, 
                                              title=f"3D Scatter Plot")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
                    st.info("Try selecting different columns or chart options.")
        
        with tab3:
            st.header("Time Series Analysis")
            
            if not datetime_cols:
                st.info("No datetime columns detected. Please ensure your data contains date/time information.")
            else:
                # Time series analysis options
                date_col = st.selectbox("Select date/time column", datetime_cols)
                value_col = st.selectbox("Select value column for time series", numeric_cols if numeric_cols else df.columns)
                
                # Convert to datetime if not already
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    # Time series chart
                    st.subheader("Time Series Plot")
                    fig = px.line(df, x=date_col, y=value_col, title=f"{value_col} over time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time aggregation options
                    st.subheader("Time Aggregation")
                    agg_options = st.radio("Aggregate by", ["Day", "Week", "Month", "Quarter", "Year"])
                    agg_func = st.selectbox("Aggregation function", ["Mean", "Sum", "Min", "Max", "Count"])
                    
                    # Map selected options to pandas functions
                    agg_map = {
                        "Day": "D",
                        "Week": "W",
                        "Month": "M",
                        "Quarter": "Q",
                        "Year": "Y"
                    }
                    
                    func_map = {
                        "Mean": "mean",
                        "Sum": "sum",
                        "Min": "min",
                        "Max": "max",
                        "Count": "count"
                    }
                    
                    # Create aggregated data
                    df_agg = df.set_index(date_col).resample(agg_map[agg_options])[value_col].agg(func_map[agg_func]).reset_index()
                    
                    # Plot aggregated data
                    fig = px.line(df_agg, x=date_col, y=value_col, 
                                 title=f"{value_col} ({agg_func}) by {agg_options}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show moving averages
                    st.subheader("Moving Averages")
                    use_ma = st.checkbox("Show moving averages")
                    
                    if use_ma:
                        ma_periods = st.multiselect("Select moving average periods", 
                                                   [3, 7, 14, 30, 60, 90], 
                                                   default=[7, 30])
                        
                        # Calculate moving averages
                        df_ts = df.set_index(date_col).sort_index()
                        fig = go.Figure()
                        
                        # Add original data
                        fig.add_trace(go.Scatter(
                            x=df_ts.index, 
                            y=df_ts[value_col],
                            mode='lines',
                            name='Original'
                        ))
                        
                        # Add moving averages
                        for period in ma_periods:
                            ma_col = f'MA-{period}'
                            df_ts[ma_col] = df_ts[value_col].rolling(window=period).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=df_ts.index,
                                y=df_ts[ma_col],
                                mode='lines',
                                name=f'{period}-period MA'
                            ))
                        
                        fig.update_layout(title=f"Moving Averages for {value_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in time series analysis: {e}")
        
        with tab4:
            st.header("Data Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Column Statistics", "Group Analysis", "Data Filtering", "Missing Values Analysis"]
            )
            
            if analysis_type == "Column Statistics":
                selected_column = st.selectbox("Select column for analysis", df.columns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Statistics for {selected_column}")
                    
                    if pd.api.types.is_numeric_dtype(df[selected_column]):
                        stats = df[selected_column].describe()
                        st.dataframe(stats)
                        
                        # Additional statistics
                        additional_stats = pd.DataFrame({
                            'Metric': ['Skewness', 'Kurtosis', 'Median', 'Mode', 'Missing Values', 'Unique Values'],
                            'Value': [
                                df[selected_column].skew(),
                                df[selected_column].kurt(),
                                df[selected_column].median(),
                                df[selected_column].mode()[0] if not df[selected_column].mode().empty else None,
                                df[selected_column].isna().sum(),
                                df[selected_column].nunique()
                            ]
                        })
                        st.dataframe(additional_stats)
                    else:
                        # For categorical columns
                        value_counts = df[selected_column].value_counts()
                        st.write("Value Counts:")
                        st.dataframe(value_counts)
                        
                        # Display missing values
                        missing = df[selected_column].isna().sum()
                        st.write(f"Missing Values: {missing} ({missing/len(df):.2%})")
                        
                        # Display unique values
                        unique = df[selected_column].nunique()
                        st.write(f"Unique Values: {unique}")
                
                with col2:
                    st.subheader(f"Visualization for {selected_column}")
                    
                    if pd.api.types.is_numeric_dtype(df[selected_column]):
                        # Distribution plot for numeric data
                        fig = px.histogram(df, x=selected_column, marginal="box", 
                                         title=f"Distribution of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Bar chart for categorical data
                        if df[selected_column].nunique() > 20:
                            top_n = st.slider("Show top N categories", 5, 20, 10)
                            top_cats = df[selected_column].value_counts().nlargest(top_n).index
                            filtered_data = df[df[selected_column].isin(top_cats)]
                            fig = px.bar(filtered_data[selected_column].value_counts().reset_index(), 
                                       x='index', y=selected_column, title=f"Top {top_n} values of {selected_column}")
                        else:
                            fig = px.bar(df[selected_column].value_counts().reset_index(), 
                                       x='index', y=selected_column, title=f"Distribution of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Group Analysis":
                # Group by analysis
                group_col = st.selectbox("Group by", categorical_cols if categorical_cols else df.columns)
                agg_col = st.selectbox("Select column to aggregate", numeric_cols if numeric_cols else df.columns)
                agg_func = st.multiselect("Select aggregation functions", 
                                        ["Mean", "Sum", "Min", "Max", "Count", "Median", "Std"],
                                        default=["Mean", "Sum", "Count"])
                
                # Map selected functions to pandas functions
                func_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Min": "min",
                    "Max": "max",
                    "Count": "count",
                    "Median": "median",
                    "Std": "std"
                }
                
                selected_funcs = [func_map[func] for func in agg_func]
                
                # Generate grouped data
                grouped_data = df.groupby(group_col)[agg_col].agg(selected_funcs).reset_index()
                
                # Display results
                st.subheader(f"Group Analysis: {agg_col} by {group_col}")
                st.dataframe(grouped_data)
                
                # Visualization
                if "mean" in selected_funcs:
                    fig = px.bar(grouped_data, x=group_col, y=f"{agg_col}_mean", 
                               title=f"Mean {agg_col} by {group_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show multiple metrics side by side
                if len(selected_funcs) > 1:
                    # Prepare data for side-by-side visualization
                    viz_data = pd.melt(grouped_data, 
                                      id_vars=[group_col], 
                                      value_vars=[f"{agg_col}_{func}" for func in selected_funcs],
                                      var_name="Metric", value_name="Value")
                    
                    # Clean up metric names
                    viz_data["Metric"] = viz_data["Metric"].str.replace(f"{agg_col}_", "")
                    
                    fig = px.bar(viz_data, x=group_col, y="Value", color="Metric", 
                               barmode="group", title=f"Multiple metrics of {agg_col} by {group_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Data Filtering":
                st.subheader("Create Custom Data Filter")
                
                # Multi-column filter builder
                filter_cols = st.multiselect("Select columns to filter by", df.columns)
                
                filtered_df = df.copy()
                
                for col in filter_cols:
                    st.subheader(f"Filter for {col}")
                    
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Numeric filter
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        filter_range = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
                        filtered_df = filtered_df[(filtered_df[col] >= filter_range[0]) & 
                                                (filtered_df[col] <= filter_range[1])]
                    else:
                        # Categorical filter
                        options = ["All"] + list(df[col].dropna().unique())
                        selected = st.multiselect(f"Values for {col}", options, default="All")
                        
                        if "All" not in selected:
                            filtered_df = filtered_df[filtered_df[col].isin(selected)]
                
                # Show filtered data
                st.subheader("Filtered Data")
                st.write(f"{len(filtered_df)} rows after filtering (out of {len(df)} total)")
                st.dataframe(filtered_df.head(100))
                
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv",
                )
            
            elif analysis_type == "Missing Values Analysis":
                st.subheader("Missing Values Analysis")
                
                # Calculate missing values stats
                missing_stats = pd.DataFrame({
                    'Missing Values': df.isna().sum(),
                    'Percentage': df.isna().mean() * 100
                }).reset_index()
                missing_stats.columns = ['Column', 'Missing Values', 'Percentage']
                missing_stats = missing_stats.sort_values('Missing Values', ascending=False)
                
                # Display stats
                st.dataframe(missing_stats)
                
                # Visualization
                fig = px.bar(missing_stats, x='Column', y='Percentage', 
                           title='Percentage of Missing Values by Column')
                st.plotly_chart(fig, use_container_width=True)
                
                # Missing values heatmap
                st.subheader("Missing Values Pattern")
                
                # Sample data if too large
                if len(df) > 1000:
                    sample_size = st.slider("Sample size for missing values visualization", 
                                           min_value=100, max_value=1000, value=500)
                    df_sample = df.sample(sample_size)
                else:
                    df_sample = df
                
                # Create missing values heatmap
                fig = px.imshow(
                    df_sample.isna().transpose(),
                    labels=dict(x="Row Index", y="Column", color="Is Missing"),
                    color_continuous_scale=["#1e88e5", "#ff0d57"],
                    title="Missing Values Pattern (White = Missing)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Options for handling missing values
                st.subheader("Handle Missing Values")
                st.write("Choose a method to handle missing values:")
                
                handling_method = st.radio(
                    "Method",
                    ["Drop rows with missing values", 
                     "Drop columns with missing values", 
                     "Fill missing values"]
                )
                
                if handling_method == "Drop rows with missing values":
                    threshold = st.slider("Drop rows with at least this many missing values:", 
                                        min_value=1, max_value=len(df.columns), value=1)
                    cleaned_df = df.dropna(thresh=len(df.columns) - threshold + 1)
                    st.write(f"Remaining rows: {len(cleaned_df)} out of {len(df)} ({len(cleaned_df)/len(df):.2%})")
                
                elif handling_method == "Drop columns with missing values":
                    threshold_pct = st.slider("Drop columns with at least this percentage of missing values:", 
                                           min_value=0, max_value=100, value=50)
                    threshold = threshold_pct / 100
                    cols_to_drop = missing_stats[missing_stats['Percentage']/100 >= threshold]['Column'].tolist()
                    cleaned_df = df.drop(columns=cols_to_drop)
                    st.write(f"Remaining columns: {len(cleaned_df.columns)} out of {len(df.columns)}")
                    st.write(f"Columns that would be dropped: {', '.join(cols_to_drop)}")
                
                elif handling_method == "Fill missing values":
                    # Select columns to fill
                    fill_cols = st.multiselect("Select columns to fill missing values", 
                                             df.columns.tolist(),
                                             default=[col for col in df.columns if df[col].isna().any()])
                    
                    cleaned_df = df.copy()
                    
                    for col in fill_cols:
                        st.subheader(f"Fill method for '{col}'")
                        
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fill_method = st.selectbox(
                                f"Method for {col}",
                                ["Mean", "Median", "Zero", "Custom value"],
                                key=f"fill_{col}"
                            )
                            
                            if fill_method == "Mean":
                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                            elif fill_method == "Median":
                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                            elif fill_method == "Zero":
                                cleaned_df[col] = cleaned_df[col].fillna(0)
                            else:  # Custom value
                                custom_val = st.number_input(f"Custom value for {col}", value=0.0)
                                cleaned_df[col] = cleaned_df[col].fillna(custom_val)
                        else:
                            fill_method = st.selectbox(
                                f"Method for {col}",
                                ["Mode (most frequent)", "Custom value", "Forward fill", "Backward fill"],
                                key=f"fill_{col}"
                            )
                            
                            if fill_method == "Mode (most frequent)":
                                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else ""
                                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                            elif fill_method == "Custom value":
                                custom_val = st.text_input(f"Custom value for {col}", value="")
                                cleaned_df[col] = cleaned_df[col].fillna(custom_val)
                            elif fill_method == "Forward fill":
                                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                            else:  # Backward fill
                                cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                
                # Show results
                st.subheader("Result after handling missing values")
                st.dataframe(cleaned_df.head(100))
                
                # Download cleaned data
                csv = cleaned_df.to_csv(index=False)
                st.download_button(
                    label="Download cleaned data as CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                )
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Please check if the Excel file is valid and not corrupted.")
else:
    # Sample data section when no file is uploaded
    st.info("👆 Upload an Excel file to get started or use the sample data below.")
    
    if st.button("Load Sample Data"):
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        data = {
            'Date': dates,
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Sales': np.random.randint(100, 1000, 100),
            'Profit': np.random.normal(300, 100, 100),
            'Customers': np.random.randint(10, 100, 100),
            'Returns': np.random.randint(0, 10, 100)
        }
        
        # Introduce some missing values
        sample_df = pd.DataFrame(data)
        for col in ['Sales', 'Profit', 'Customers']:
            mask = np.random.random(len(sample_df)) < 0.05
            sample_df.loc[mask, col] = np.nan
            
        # Cache the sample dataframe
        st.session_state['sample_df'] = sample_df
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(sample_df.head(10))
        
        # Create download button for sample data
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download sample data as Excel",
            data=csv,
            file_name="sample_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Add imports for plotly that were missing above
import plotly.graph_objects as go

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use this app:
1. Upload your Excel file using the file uploader at the top
2. Navigate through the tabs to explore your data and create visualizations
3. Customize the visualizations using the controls on the left side
4. Download results as needed

Built with Streamlit and Python data visualization libraries.
""")
