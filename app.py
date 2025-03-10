import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from supervised import AutoML
import numpy as np
import os
import time

# Set Page Config
st.set_page_config(layout="wide", page_title="Advanced Data Analysis Dashboard")

# Initialize session state for model tracking
if 'automl_model' not in st.session_state:
    st.session_state.automl_model = None
if 'model_columns' not in st.session_state:
    st.session_state.model_columns = None

st.title("Advanced Data Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Traditional ML", "AutoML", "Predictions"])

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Ensure only numeric columns are used for numeric operations
    df_numeric = df.select_dtypes(include=['number'])
    
    # DATA OVERVIEW PAGE
    if page == "Data Overview":
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Data Preview")
            st.write(df.head())
        
        with col2:
            st.write("### Data Information")
            buffer = []
            buffer.append(f"Number of rows: {df.shape[0]}")
            buffer.append(f"Number of columns: {df.shape[1]}")
            buffer.append("Column names and types:")
            for col in df.columns:
                buffer.append(f"  - {col}: {df[col].dtype}")
            st.text("\n".join(buffer))
            
            st.write("### Missing Values")
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing Values']
            missing_data['Missing %'] = round(missing_data['Missing Values'] / len(df) * 100, 2)
            st.write(missing_data)
        
        st.write("### Data Summary Statistics")
        st.write(df.describe())
        
        st.write("### Unique Values in Categorical Columns")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            selected_cat_col = st.selectbox("Select categorical column", cat_cols)
            st.write(df[selected_cat_col].value_counts())
            
            # Pie chart for categorical data
            fig = px.pie(values=df[selected_cat_col].value_counts().values, 
                          names=df[selected_cat_col].value_counts().index, 
                          title=f"Distribution of {selected_cat_col}")
            st.plotly_chart(fig)
    
    # EXPLORATORY ANALYSIS PAGE
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlation", "Distributions", "Pairplots", "Filtering", "Grouping"])
        
        with tab1:
            st.write("### Correlation Heatmap")
            st.write("The correlation heatmap represents the relationships between numerical features. A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation.")
            fig = px.imshow(df_numeric.corr(), text_auto=True, color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.write("### Distribution Analysis")
            
            # Histogram for Numeric Columns
            st.write("#### Histogram")
            selected_hist_col = st.selectbox("Select a column for Histogram", df_numeric.columns)
            fig = px.histogram(df, x=selected_hist_col, marginal="box", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            
            # Box Plot
            st.write("#### Box Plot")
            selected_box_col = st.selectbox("Select column for Box Plot", df_numeric.columns, key="box_col")
            selected_group_col = st.selectbox("Group by (optional)", ["None"] + list(df.columns))
            
            if selected_group_col == "None":
                fig = px.box(df, y=selected_box_col)
            else:
                fig = px.box(df, x=selected_group_col, y=selected_box_col)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.write("### Pairplot")
            selected_pairplot_cols = st.multiselect("Select columns for Pairplot", df_numeric.columns, df_numeric.columns[:3] if len(df_numeric.columns) >= 3 else df_numeric.columns)
            if len(selected_pairplot_cols) >= 2:
                st.plotly_chart(px.scatter_matrix(df_numeric[selected_pairplot_cols]), use_container_width=True)
            else:
                st.write("Select at least two columns for the pairplot.")
        
        with tab4:
            st.write("### Interactive Filtering")
            filter_col = st.selectbox("Select column to filter", df.columns)
            
            if df[filter_col].dtype == "object" or df[filter_col].nunique() < 10:
                # For categorical columns or columns with few unique values
                unique_vals = df[filter_col].unique()
                selected_vals = st.multiselect("Select values", unique_vals, unique_vals[:3] if len(unique_vals) >= 3 else unique_vals)
                if selected_vals:
                    filtered_df = df[df[filter_col].isin(selected_vals)]
                    st.write(filtered_df)
                else:
                    st.write("Please select at least one value to filter.")
            else:
                # For numerical columns
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                range_vals = st.slider(f"Select range for {filter_col}", min_val, max_val, (min_val, max_val))
                filtered_df = df[(df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])]
                st.write(filtered_df)
        
        with tab5:
            st.write("### Grouping and Aggregation")
            group_col = st.selectbox("Select column to group by", df.columns)
            agg_col = st.selectbox("Select column to aggregate", df_numeric.columns)
            agg_func = st.selectbox("Select aggregation function", ["mean", "median", "sum", "count", "min", "max"])
            
            if agg_func == "mean":
                grouped_df = df.groupby(group_col)[agg_col].mean().reset_index()
            elif agg_func == "median":
                grouped_df = df.groupby(group_col)[agg_col].median().reset_index()
            elif agg_func == "sum":
                grouped_df = df.groupby(group_col)[agg_col].sum().reset_index()
            elif agg_func == "count":
                grouped_df = df.groupby(group_col)[agg_col].count().reset_index()
            elif agg_func == "min":
                grouped_df = df.groupby(group_col)[agg_col].min().reset_index()
            else:
                grouped_df = df.groupby(group_col)[agg_col].max().reset_index()
            
            st.write(grouped_df)
            
            # Bar chart for grouped data
            fig = px.bar(grouped_df, x=group_col, y=agg_col, title=f"{agg_func.capitalize()} of {agg_col} by {group_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    # TRADITIONAL ML PAGE
    elif page == "Traditional ML":
        st.header("Traditional Machine Learning")
        
        tab1, tab2, tab3 = st.tabs(["Regression", "Classification", "Clustering"])
        
        with tab1:
            st.write("### Regression Model Training")
            st.write("This section allows you to train regression models and evaluate their performance.")
            
            target = st.selectbox("Select target column for regression", df_numeric.columns)
            features = st.multiselect("Select features for regression", 
                                     [col for col in df_numeric.columns if col != target],
                                     default=[col for col in df_numeric.columns if col != target][:3] if len(df_numeric.columns) > 3 else [col for col in df_numeric.columns if col != target])
            
            if not features:
                st.warning("Please select at least one feature.")
            else:
                test_size = st.slider("Test size (%)", 10, 50, 20) / 100
                X_train, X_test, y_train, y_test = train_test_split(df_numeric[features], df_numeric[target], test_size=test_size, random_state=42)
                
                model_choice = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regressor"])
                
                if st.button("Train Regression Model"):
                    with st.spinner("Training model..."):
                        if model_choice == "Linear Regression":
                            model = LinearRegression()
                            # Simple param grid for linear regression
                            param_grid = {'fit_intercept': [True, False]}
                        else:
                            model = RandomForestRegressor(random_state=42)
                            # More complex param grid for random forest
                            param_grid = {
                                'n_estimators': [50, 100],
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5]
                            }
                        
                        # Use GridSearchCV for hyperparameter tuning
                        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        
                        # Make predictions and evaluate
                        y_pred = best_model.predict(X_test)
                        
                        # Display metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = best_model.score(X_test, y_test)
                        
                        st.write(f"**Best Parameters:** {grid_search.best_params_}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MAE", f"{mae:.4f}")
                        col2.metric("MSE", f"{mse:.4f}")
                        col3.metric("RMSE", f"{rmse:.4f}")
                        col4.metric("R²", f"{r2:.4f}")
                        
                        # Plot actual vs predicted
                        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                        fig.add_shape(type='line', x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                                     line=dict(color='red', dash='dash'))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance for Random Forest
                        if model_choice == "Random Forest Regressor":
                            st.write("### Feature Importance")
                            feature_importance = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
                            fig = px.bar(x=feature_importance.index, y=feature_importance.values, title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Store model in session state
                        st.session_state.reg_model = best_model
                        st.session_state.reg_features = features
        
        with tab2:
            st.write("### Classification Model Training")
            
            # Check if there are categorical columns that could be targets
            potential_class_targets = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    potential_class_targets.append(col)
            
            if not potential_class_targets:
                st.warning("No suitable categorical columns found for classification. Consider creating a categorical target.")
            else:
                target = st.selectbox("Select target column for classification", potential_class_targets)
                
                # Convert target to categorical if not already
                df['target_class'] = df[target].astype('category').cat.codes
                
                # Select features
                features = st.multiselect("Select features for classification", 
                                         [col for col in df_numeric.columns if col != 'target_class'],
                                         default=[col for col in df_numeric.columns if col != 'target_class'][:3] if len(df_numeric.columns) > 3 else [col for col in df_numeric.columns if col != 'target_class'])
                
                if not features:
                    st.warning("Please select at least one feature.")
                else:
                    test_size = st.slider("Test size (%) for classification", 10, 50, 20, key="class_test_size") / 100
                    X_train, X_test, y_train, y_test = train_test_split(df[features], df['target_class'], test_size=test_size, random_state=42)
                    
                    if st.button("Train Classification Model"):
                        with st.spinner("Training classification model..."):
                            model = RandomForestClassifier(random_state=42)
                            param_grid = {
                                'n_estimators': [50, 100],
                                'max_depth': [None, 10, 20]
                            }
                            
                            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
                            grid_search.fit(X_train, y_train)
                            best_model = grid_search.best_estimator_
                            
                            # Make predictions and evaluate
                            y_pred = best_model.predict(X_test)
                            
                            # Display metrics
                            accuracy = best_model.score(X_test, y_test)
                            
                            st.write(f"**Best Parameters:** {grid_search.best_params_}")
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            
                            # Confusion Matrix
                            st.write("### Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Convert labels back to original categories
                            labels = df[target].astype('category').cat.categories
                            
                            fig = px.imshow(cm, 
                                           labels=dict(x="Predicted", y="Actual"), 
                                           x=[str(i) for i in range(len(labels))], 
                                           y=[str(i) for i in range(len(labels))],
                                           text_auto=True)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance
                            st.write("### Feature Importance")
                            feature_importance = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
                            fig = px.bar(x=feature_importance.index, y=feature_importance.values, title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store model in session state
                            st.session_state.class_model = best_model
                            st.session_state.class_features = features
        
        with tab3:
            st.write("### K-Means Clustering")
            st.write("K-Means Clustering groups similar data points together based on selected features.")
            
            cluster_col = st.multiselect("Select columns for clustering", df_numeric.columns, df_numeric.columns[:2] if len(df_numeric.columns) >= 2 else df_numeric.columns)
            
            if len(cluster_col) < 2:
                st.warning("Select at least two columns for clustering.")
            else:
                # Use the Elbow Method to find optimal K
                st.write("#### Elbow Method for Optimal K")
                max_k = min(10, df.shape[0] // 5)  # Limit max K
                inertia = []
                k_range = range(1, max_k + 1)
                
                # Calculate inertia for different K values
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(df[cluster_col])
                    inertia.append(kmeans.inertia_)
                
                # Plot Elbow Method
                fig = px.line(x=list(k_range), y=inertia, markers=True, 
                             labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'})
                st.plotly_chart(fig, use_container_width=True)
                
                # K selection
                num_clusters = st.slider("Select number of clusters (K)", 2, max_k, 3)
                
                if st.button("Run Clustering"):
                    with st.spinner("Clustering data..."):
                        # Run K-Means
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(df[cluster_col])
                        
                        # Add clusters to dataframe
                        temp_df = df.copy()
                        temp_df['Cluster'] = clusters
                        
                        # Plot Clusters
                        if len(cluster_col) == 2:
                            fig = px.scatter(temp_df, x=cluster_col[0], y=cluster_col[1], 
                                           color='Cluster', hover_data=df.columns)
                            st.plotly_chart(fig, use_container_width=True)
                        elif len(cluster_col) >= 3:
                            fig = px.scatter_3d(temp_df, x=cluster_col[0], y=cluster_col[1], z=cluster_col[2],
                                              color='Cluster', hover_data=df.columns)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Cluster Analysis
                        st.write("### Cluster Analysis")
                        for i in range(num_clusters):
                            with st.expander(f"Cluster {i} Analysis"):
                                cluster_data = temp_df[temp_df['Cluster'] == i]
                                st.write(f"Number of samples: {len(cluster_data)}")
                                st.write(cluster_data.describe())
    
    # AUTOML PAGE
    elif page == "AutoML":
        st.header("AutoML with mljar-supervised")
        
        # Create a results directory
        results_path = "./automl_results"
        os.makedirs(results_path, exist_ok=True)
        
        # Target selection
        st.write("### Select Target and Settings")
        target_column = st.selectbox("Select the target column", df.columns)
        
        # Determine task type
        task_options = ["regression", "binary_classification", "multiclass_classification"]
        
        # Smart default selection
        if df[target_column].nunique() == 2:
            default_task = "binary_classification"
        elif df[target_column].nunique() > 2 and df[target_column].dtype == "object":
            default_task = "multiclass_classification"
        else:
            default_task = "regression"
        
        ml_task = st.selectbox("Select ML task", task_options, index=task_options.index(default_task))
        
        # Validate ML task
        validation_failed = False
        if ml_task == "binary_classification" and df[target_column].nunique() != 2:
            st.error("Error: The target column must have exactly 2 unique values for binary classification.")
            validation_failed = True
        elif ml_task == "multiclass_classification" and df[target_column].nunique() <= 2:
            st.error("Error: The target column must have more than 2 unique values for multiclass classification.")
            validation_failed = True
        elif ml_task == "regression" and not pd.api.types.is_numeric_dtype(df[target_column]):
            st.error("Error: The target column must contain numeric values for regression.")
            validation_failed = True
        
        # AutoML settings
        col1, col2 = st.columns(2)
        with col1:
            mode = st.selectbox(
                "Select mode",
                ["Explain", "Perform", "Compete", "Optuna"],
                help="Explain: For data understanding, Perform: For production, Compete: For competitions, Optuna: For hyperparameter tuning",
            )
            explain_level = st.slider(
                "Explain level",
                min_value=0,
                max_value=2,
                value=2,
                help="0: No explanations, 1: Basic explanations, 2: Detailed explanations",
            )
        
        with col2:
            time_limit = st.number_input(
                "Time limit (seconds)",
                value=600,
                min_value=60,
                help="Maximum time for AutoML training",
            )
            algorithms = st.multiselect(
                "Select algorithms to include",
                ["Linear", "Random Forest", "Extra Trees", "LightGBM", "Xgboost", "CatBoost", "Neural Network"],
                default=["Linear", "Random Forest", "LightGBM"]
            )
        
        # Train button
        if st.button("Train AutoML Model") and not validation_failed:
            with st.spinner("AutoML training in progress... This might take a while."):
                try:
                    # Start timer
                    start_time = time.time()
                    
                    # Prepare data
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    
                    # Store column names for later validation
                    st.session_state.model_columns = list(X.columns)
                    
                    # Map algorithms to AutoML format
                    algorithms_mapping = {
                        "Linear": "Linear",
                        "Random Forest": "Random Forest",
                        "Extra Trees": "Extra Trees",
                        "LightGBM": "LightGBM",
                        "Xgboost": "Xgboost",
                        "CatBoost": "CatBoost",
                        "Neural Network": "Neural Network"
                    }
                    selected_algorithms = [algorithms_mapping[algo] for algo in algorithms]
                    
                    # Initialize AutoML
                    automl = AutoML(
                        results_path=results_path,
                        mode=mode,
                        ml_task=ml_task,
                        algorithms=selected_algorithms,
                        total_time_limit=time_limit,
                        explain_level=explain_level,
                        random_state=42,
                    )
                    
                    # Train the model
                    automl.fit(X, y)
                    
                    # Save the model to session state
                    st.session_state.automl_model = automl
                    
                    # End timer
                    elapsed_time = time.time() - start_time
                    
                    # Display results
                    st.success(f"Training completed in {elapsed_time:.2f} seconds!")
                    
                    # Try to display leaderboard
                    try:
                        # For newer versions
                        leaderboard = automl.leaderboard()
                        st.write("### AutoML Leaderboard")
                        st.dataframe(leaderboard)
                    except:
                        try:
                            # For older versions - read CSV
                            leaderboard_path = os.path.join(results_path, "leaderboard.csv")
                            if os.path.exists(leaderboard_path):
                                leaderboard = pd.read_csv(leaderboard_path)
                                st.write("### AutoML Leaderboard")
                                st.dataframe(leaderboard)
                        except:
                            st.write("Leaderboard information not available.")
                    
                    # Display feature importance if available
                    try:
                        for root, dirs, files in os.walk(results_path):
                            importance_files = [f for f in files if f.endswith(".png") and "importance" in f]
                            if importance_files:
                                st.write("### Feature Importance")
                                for imp_file in importance_files[:1]:  # Show only the first one
                                    st.image(os.path.join(root, imp_file))
                                    break
                    except:
                        st.info("Feature importance visualization not available.")
                    
                    # Try to generate and display predictions
                    try:
                        predictions = automl.predict(X)
                        
                        if ml_task in ["binary_classification", "multiclass_classification"]:
                            st.write("### Confusion Matrix (Training Data)")
                            cm = confusion_matrix(y, predictions)
                            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                          labels=dict(x="Predicted", y="Actual"))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if ml_task == "regression":
                            st.write("### Predicted vs Actual (Training Data)")
                            fig = px.scatter(x=y, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'})
                            fig.add_shape(type='line', x0=min(y), y0=min(y), x1=max(y), y1=max(y),
                                         line=dict(color='red', dash='dash'))
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate predictions on training data: {str(e)}")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Check that you have the mljar-supervised package installed: pip install mljar-supervised")
                    
                    # Try to get version info
                    try:
                        import supervised
                        st.info(f"Current mljar-supervised version: {supervised.__version__}")
                    except:
                        st.info("Could not determine mljar-supervised version.")
        
        # Display any existing results
        if 'automl_model' in st.session_state and st.session_state.automl_model is not None:
            st.write("### Current AutoML Model")
            st.success("A trained AutoML model is loaded and ready for predictions.")
    
    # PREDICTIONS PAGE
    elif page == "Predictions":
        st.header("Make Predictions")
        
        prediction_tabs = st.tabs(["Manual Input", "File Upload", "Sample Predictions"])
        
        with prediction_tabs[0]:
            st.write("### Manual Input for Prediction")
            
            if 'reg_model' in st.session_state or 'class_model' in st.session_state or 'automl_model' in st.session_state:
                # Determine which models are available
                available_models = []
                if 'reg_model' in st.session_state:
                    available_models.append("Regression Model")
                if 'class_model' in st.session_state:
                    available_models.append("Classification Model")
                if 'automl_model' in st.session_state:
                    available_models.append("AutoML Model")
                
                # Let user select a model
                selected_model = st.selectbox("Select model for prediction", available_models)
                
                if selected_model == "Regression Model" and 'reg_model' in st.session_state:
                    features = st.session_state.reg_features
                    user_input = {}
                    
                    st.write("Enter feature values:")
                    col1, col2 = st.columns(2)
                    for i, feature in enumerate(features):
                        if i % 2 == 0:
                            with col1:
                                user_input[feature] = st.number_input(
                                    f"{feature}", 
                                    value=float(df[feature].mean()),
                                    key=f"reg_{feature}"
                                )
                        else:
                            with col2:
                                user_input[feature] = st.number_input(
                                    f"{feature}", 
                                    value=float(df[feature].mean()),
                                    key=f"reg_{feature}"
                                )
                    
                    if st.button("Predict (Regression)"):
                        input_df = pd.DataFrame([user_input])
                        prediction = st.session_state.reg_model.predict(input_df)
                        st.success(f"Predicted Value: {prediction[0]:.4f}")
                
                elif selected_model == "Classification Model" and 'class_model' in st.session_state:
                    features = st.session_state.class_features
                    user_input = {}
                    
                    st.write("Enter feature values:")
                    col1, col2 = st.columns(2)
                    for i, feature in enumerate(features):
                        if i % 2 == 0:
                            with col1:
                                user_input[feature] = st.number_input(
                                    f"{feature}", 
                                    value=float(df[feature].mean()),
                                    key=f"class_{feature}"
                                )
                        else:
                            with col2:
                                user_input[feature] = st.number_input(
                                    f"{feature}", 
                                    value=float(df[feature].mean()),
                                    key=f"class_{feature}"
                                )
                    
                    if st.button("Predict (Classification)"):
                        input_df = pd.DataFrame([user_input])
                        prediction = st.session_state.class_model.predict(input_df)
                        st.success(f"Predicted Class: {prediction[0]}")
                
                elif selected_model == "AutoML Model" and 'automl_model' in st.session_state:
                    # Get all features needed for AutoML model
                    if st.session_state.model_columns:
                        features = st.session_state.model_columns
                        user_input = {}
                        
                        st.write("Enter feature values:")
                        col1, col2 = st.columns(2)
                        for i, feature in enumerate(features):
                            if i % 2 == 0:
                                with col1:
                                    user_input[feature] = st.number_input(
                                        f"{feature}", 
                                        value=float(df[feature].mean()),
                                        key=f"auto_{feature}"
                                    )
                            else:
                                with col2:
                                    user_input[feature] = st.number_input(
                                        f"{feature}", 
                                        value=float(df[feature].mean()),
                                        key=f"auto_{feature}"
                                    )

                    if st.button("Predict (AutoML)"):
                            try:
                                input_df = pd.DataFrame([user_input])
                                prediction = st.session_state.automl_model.predict(input_df)
                                
                                # Handle different prediction types
                                if isinstance(prediction, np.ndarray):
                                    if len(prediction.shape) == 1:
                                        st.success(f"Predicted Value: {prediction[0]}")
                                    else:
                                        st.success(f"Prediction: {prediction[0]}")
                                else:
                                    st.success(f"Prediction: {prediction}")
                                
                                # If probability predictions are available (for classification)
                                try:
                                    if hasattr(st.session_state.automl_model, 'predict_proba'):
                                        proba = st.session_state.automl_model.predict_proba(input_df)
                                        st.write("### Prediction Probabilities")
                                        
                                        # Format as dataframe for display
                                        if isinstance(proba, np.ndarray):
                                            if len(proba.shape) > 1 and proba.shape[1] > 1:
                                                proba_df = pd.DataFrame(proba[0])
                                                proba_df.columns = [f"Class {i}" for i in range(proba_df.shape[1])]
                                                st.dataframe(proba_df.T)
                                                
                                                # Show bar chart of probabilities
                                                fig = px.bar(
                                                    x=[f"Class {i}" for i in range(proba.shape[1])],
                                                    y=proba[0],
                                                    title="Prediction Probabilities"
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.info(f"Probability information not available: {str(e)}")
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                                st.info("Make sure the input features match the features used during training.")
                    else:
                        st.warning("Feature information for AutoML model not available.")
            else:
                st.warning("No trained models available. Please train a model in the respective tabs first.")

        with prediction_tabs[1]:
            st.write("### Predict from File")
            
            pred_file = st.file_uploader("Upload file for prediction", type=["csv"])
            if pred_file is not None:
                pred_df = pd.read_csv(pred_file)
                st.write("Preview of uploaded data:")
                st.write(pred_df.head())
                
                if 'automl_model' in st.session_state:
                    if st.button("Generate Predictions"):
                        try:
                            # Check for required columns
                            missing_cols = [col for col in st.session_state.model_columns if col not in pred_df.columns]
                            if missing_cols:
                                st.error(f"Missing columns in prediction file: {', '.join(missing_cols)}")
                            else:
                                # Make predictions
                                with st.spinner("Generating predictions..."):
                                    predictions = st.session_state.automl_model.predict(pred_df[st.session_state.model_columns])
                                    
                                    # Add predictions to dataframe
                                    results_df = pred_df.copy()
                                    results_df['Prediction'] = predictions
                                    
                                    # Attempt to get probabilities for classification
                                    try:
                                        if hasattr(st.session_state.automl_model, 'predict_proba'):
                                            probas = st.session_state.automl_model.predict_proba(pred_df[st.session_state.model_columns])
                                            if isinstance(probas, np.ndarray) and len(probas.shape) > 1 and probas.shape[1] > 1:
                                                for i in range(probas.shape[1]):
                                                    results_df[f'Probability_Class_{i}'] = probas[:, i]
                                    except Exception as e:
                                        st.info(f"Probability information could not be added: {str(e)}")
                                    
                                    st.write("### Prediction Results")
                                    st.dataframe(results_df)
                                    
                                    # Download button for predictions
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Predictions as CSV",
                                        data=csv,
                                        file_name="predictions.csv",
                                        mime="text/csv",
                                    )
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                else:
                    st.warning("Please train an AutoML model first before making batch predictions.")
        
        with prediction_tabs[2]:
            st.write("### Sample Predictions")
            
            if 'automl_model' in st.session_state:
                # Select a random sample from the dataset
                sample_size = st.slider("Number of random samples", 1, 20, 5)
                if st.button("Generate Sample Predictions"):
                    if st.session_state.model_columns:
                        with st.spinner("Generating sample predictions..."):
                            # Get random samples
                            sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
                            samples = df.loc[sample_indices]
                            
                            # Separate features and target if present
                            features_sample = samples[st.session_state.model_columns]
                            
                            # Make predictions
                            predictions = st.session_state.automl_model.predict(features_sample)
                            
                            # Create results dataframe
                            results_df = samples.copy()
                            results_df['Prediction'] = predictions
                            
                            # If we have the original target, compare prediction to actual
                            target_col = [col for col in df.columns if col not in st.session_state.model_columns]
                            if target_col:
                                st.write("### Sample Predictions vs Actual")
                                st.dataframe(results_df)
                                
                                # For regression, show actual vs predicted
                                if isinstance(predictions[0], (int, float, np.number)):
                                    fig = px.scatter(
                                        results_df, 
                                        x=target_col[0], 
                                        y='Prediction',
                                        labels={'x': 'Actual', 'y': 'Predicted'}
                                    )
                                    fig.add_shape(
                                        type='line',
                                        x0=min(results_df[target_col[0]]),
                                        y0=min(results_df[target_col[0]]),
                                        x1=max(results_df[target_col[0]]),
                                        y1=max(results_df[target_col[0]]),
                                        line=dict(color='red', dash='dash')
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("### Sample Predictions")
                                st.dataframe(results_df)
                    else:
                        st.warning("Feature information not available. Please retrain the model.")
            else:
                st.warning("No AutoML model available. Please train a model first.")

else:
    st.info("Please upload a CSV file to get started.")
    
    # Display sample datasets that can be used
    st.write("### Sample Datasets You Can Try")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Regression**")
        st.write("- Boston Housing")
        st.write("- California Housing")
        st.write("- Diabetes")
    
    with col2:
        st.write("**Classification**")
        st.write("- Iris Flower")
        st.write("- Breast Cancer")
        st.write("- Wine Quality")
    
    with col3:
        st.write("**Time Series**")
        st.write("- Air Quality")
        st.write("- Stock Prices")
        st.write("- Energy Consumption")
    
    # Add a sample download option
    st.write("### Sample File Format")
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 10),
        'feature2': np.random.normal(5, 2, 10),
        'feature3': np.random.choice(['A', 'B', 'C'], 10),
        'target': np.random.randint(0, 2, 10)
    })
    
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_data.csv",
        mime="text/csv",
    )
    
    # Instructions
    with st.expander("How to Use This Dashboard"):
        st.write("""
        ### Quick Start Guide
        
        1. **Upload Data**: Start by uploading a CSV file using the uploader in the sidebar.
        
        2. **Explore Your Data**: Use the 'Data Overview' and 'Exploratory Analysis' tabs to understand your dataset.
        
        3. **Train Models**: 
           - Use 'Traditional ML' for standard regression, classification, or clustering.
           - Use 'AutoML' for automated model selection and hyperparameter tuning.
        
        4. **Make Predictions**: Once models are trained, use the 'Predictions' tab to:
           - Make individual predictions with manual input
           - Generate batch predictions from a file
           - See example predictions on sample data
        
        ### Features
        
        - **Data Overview**: View basic information, missing values, and summary statistics
        - **Exploratory Analysis**: Create visualizations, filter data, and perform grouping
        - **Traditional ML**: Train regression, classification, and clustering models
        - **AutoML**: Automatically find the best model with minimal effort
        - **Predictions**: Apply your models to new data
        """)
