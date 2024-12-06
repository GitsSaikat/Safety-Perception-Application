import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap

# Load Dataset
data_path = 'Survey Final.csv'
df = pd.read_csv(data_path)

# Encode Target Column
le = LabelEncoder()
df['Percieved Safety'] = le.fit_transform(df['Percieved Safety'])

# Data Splitting (Global for Use in All Sections)
test_size = 0.2  # Default test size (can be changed in data splitting section)
X = df.drop(columns=['Percieved Safety'])
y = df['Percieved Safety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Streamlit App
st.set_page_config(page_title="Evaluating Safety Perception on Commuting App", layout='wide')
st.title("Evaluating Safety Perception on Commuting ")

# Sidebar section
with st.sidebar:
    st.image("logo.png", use_container_width=True, caption="Safety Perception")
    st.markdown("---") 
    selected = st.selectbox(
        "Navigation",
        [
            "📊 Data Overview",
            "🔍 Exploratory Data Analysis",
            "🤖 Model Training, Evaluation & Explanations",
            "🔮 Predict Perceived Safety"
        ]
    )

# Data Overview
if selected == "📊 Data Overview":
    st.header("📊 Data Overview")
    if st.checkbox("Show Dataset"):
        st.write(df.head())
        st.write(f"Dataset Shape: {df.shape}")
        st.write("Data Types:")
        st.write(df.dtypes)

# Exploratory Data Analysis
if selected == "🔍 Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    if st.checkbox("Correlation Heatmap"):
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if st.checkbox("Histogram"):
        st.write("Histograms of Numeric Columns")
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_column = st.selectbox("Select Column for Histogram", numeric_columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], kde=True, ax=ax)
        st.pyplot(fig)

    if st.checkbox("Boxplot for Numeric Columns"):
        st.write("Boxplot of Numeric Columns")
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_column = st.selectbox("Select Column for Boxplot", numeric_columns)
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=selected_column, ax=ax)
        st.pyplot(fig)

    if st.checkbox("Pairplot of Dataset"):
        st.write("Pairplot of the Dataset")
        fig = sns.pairplot(df)
        st.pyplot(fig)

# Model Training, Evaluation & Explanations
if selected == "🤖 Model Training, Evaluation & Explanations":
    st.header("🤖 Model Training, Evaluation & Explanations")
    if st.checkbox("Train, Evaluate, and Explain Models"):
        # Model Training
        st.write("Training Tree-Based Models")
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "Histogram Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
        }
        
        model_preds = {}
        model_accuracies = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            model_preds[model_name] = preds
            model_accuracies[model_name] = accuracy
            st.write(f"{model_name} Accuracy: {accuracy:.2f}")
        
        # Model Evaluation
        selected_model = st.selectbox("Select Model for Detailed Evaluation", list(models.keys()))
        selected_model_instance = models[selected_model]
        selected_preds = model_preds[selected_model]
        st.write("Classification Report:")
        st.text(classification_report(y_test, selected_preds))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, selected_preds))
        
        # Feature Importance
        if st.checkbox("Show Feature Importance"):
            st.write(f"Feature Importance from {selected_model} Model")
            if hasattr(selected_model_instance, 'feature_importances_'):
                feature_importances = selected_model_instance.feature_importances_
                importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": feature_importances})
                importance_df = importance_df.sort_values(by="Importance", ascending=False)
                st.bar_chart(importance_df.set_index("Feature"))
            else:
                st.write("The selected model does not support feature importances.")
        
        # SHAP Explanations
        if st.checkbox("Explain Predictions with SHAP"):
            st.write(f"SHAP Explanation for {selected_model} Model")
            explainer = shap.TreeExplainer(selected_model_instance)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            st.pyplot()

# Predict Percieved Safety
if selected == "🔮 Predict Perceived Safety":
    st.header("🔮 Predict Percieved Safety")
    st.write("Please provide the following information to predict Percieved Safety for transport:")
    
    # User Input for Prediction
    overcrowding = st.selectbox("How overcrowded do you think the transport is on a scale from 0 (Not overcrowded) to 4 (Very overcrowded)?", [0, 1, 2, 3, 4])
    preference = st.selectbox("How much do you prefer this mode of transport on a scale from 0 (Not preferred) to 4 (Highly preferred)?", [0, 1, 2, 3, 4])
    daytime_safety = st.selectbox("How safe do you feel using this transport during the daytime on a scale from 0 (Not safe) to 4 (Very safe)?", [0, 1, 2, 3, 4])
    nighttime_safety = st.selectbox("How safe do you feel using this transport during the nighttime on a scale from 0 (Not safe) to 4 (Very safe)?", [0, 1, 2, 3, 4])
    taxi_dsafety = st.selectbox("How safe do you feel using a taxi during the day on a scale from 0 (Not safe) to 4 (Very safe)?", [0, 1, 2, 3, 4])
    taxi_nsafety = st.selectbox("How safe do you feel using a taxi during the night on a scale from 0 (Not safe) to 4 (Very safe)?", [0, 1, 2, 3, 4])
    reporting = st.selectbox("How comfortable are you with reporting incidents related to this transport on a scale from 0 (Not comfortable) to 4 (Very comfortable)?", [0, 1, 2, 3, 4])
    background_check = st.selectbox("How effective do you think background checks are for transport personnel on a scale from 0 (Not effective) to 4 (Very effective)?", [0, 1, 2, 3, 4])

    user_data = np.array([[
        overcrowding, preference, daytime_safety, nighttime_safety,
        taxi_dsafety, taxi_nsafety, reporting, background_check
    ]])
    
    if st.button("Predict Percieved Safety"):
        # Train the Model (Again) and Predict
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        prediction = model.predict(user_data)
        predicted_class = le.inverse_transform(prediction)
        
        st.write(f"Predicted Percieved Safety Class: {predicted_class[0]}")
