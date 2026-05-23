# %%
# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# %%
# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    layout="wide"
)

st.title("📊 AI-Powered Predictive Analytics Dashboard")

st.markdown("---")

# %%
# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("sales_data.csv")

# Convert date column
df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    dayfirst=True
)

st.success("Dataset Loaded Successfully")

# %%
# ============================================
# DATA PREVIEW
# ============================================

st.subheader("📄 Dataset Preview")

st.dataframe(df.head())

# %%
# ============================================
# DATA INFORMATION
# ============================================

st.subheader("📌 Dataset Information")

st.write(df.describe())

# %%
# ============================================
# NULL VALUES
# ============================================

st.subheader("🧹 Null Values")

st.write(df.isnull().sum())

# %%
# ============================================
# REMOVE NULL VALUES & DUPLICATES
# ============================================

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

# %%
# ============================================
# FEATURE ENGINEERING
# ============================================

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

# %%
# ============================================
# SALES TREND GRAPH
# ============================================

st.subheader("📈 Sales Trend Over Time")

line_fig = px.line(
    df,
    x='Order Date',
    y='Sales',
    title='Sales Trend Analysis'
)

st.plotly_chart(line_fig, use_container_width=True)

# %%
# ============================================
# MONTHLY SALES ANALYSIS
# ============================================

monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()

st.subheader("📊 Monthly Sales Analysis")

bar_fig = px.bar(
    monthly_sales,
    x='Month',
    y='Sales',
    title='Monthly Sales'
)

st.plotly_chart(bar_fig, use_container_width=True)

# %%
# ============================================
# SALES DISTRIBUTION
# ============================================

st.subheader("📉 Sales Distribution")

hist_fig = px.histogram(
    df,
    x='Sales',
    nbins=30,
    title='Sales Distribution'
)

st.plotly_chart(hist_fig, use_container_width=True)

# %%
# ============================================
# CORRELATION HEATMAP
# ============================================

# %%
# ============================================
# CORRELATION HEATMAP
# ============================================

st.subheader("🔥 Correlation Heatmap")

# Select only numeric columns
numeric_df = df.select_dtypes(include=np.number)

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax
)

st.pyplot(fig)
# %%
# ============================================
# FEATURES & TARGET
# ============================================

X = df[['Year', 'Month', 'Day']]

y = df['Sales']

# %%
# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
# ============================================
# TRAIN MODEL
# ============================================

model = LinearRegression()

model.fit(X_train, y_train)

# %%
# ============================================
# SAVE MODEL
# ============================================

joblib.dump(model, "model.pkl")

# %%
# ============================================
# MAKE PREDICTIONS
# ============================================

predictions = model.predict(X_test)

# %%
# ============================================
# MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

st.subheader("🤖 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.2f}")

    st.metric("MSE", f"{mse:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

    st.metric("R² Score", f"{r2:.2f}")

# %%
# ============================================
# ACTUAL VS PREDICTED GRAPH
# ============================================

st.subheader("🎯 Actual vs Predicted Sales")

scatter_fig = px.scatter(
    x=y_test,
    y=predictions,
    labels={
        'x': 'Actual Sales',
        'y': 'Predicted Sales'
    },
    title='Actual vs Predicted'
)

st.plotly_chart(scatter_fig, use_container_width=True)

# %%
# ============================================
# FUTURE SALES PREDICTION
# ============================================

st.subheader("🔮 Predict Future Sales")

selected_date = st.date_input(
    "Select Future Date"
)

year = selected_date.year
month = selected_date.month
day = selected_date.day

future_prediction = model.predict(
    [[year, month, day]]
)

st.success(
    f"Predicted Sales for {selected_date}: ₹ {future_prediction[0]:.2f}"
)

# %%
# ============================================
# TOP SALES RECORDS
# ============================================

st.subheader("🏆 Top 10 Sales Records")

top_sales = df.sort_values(
    by='Sales',
    ascending=False
).head(10)

st.dataframe(top_sales)

# %%
# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown(
    "### ✅ Predictive Analytics Using Historical Data"
)

st.markdown(
    """
This project demonstrates:
- Predictive Modeling
- Data Cleaning
- Regression Analysis
- Trend Forecasting
- Data Visualization
- Machine Learning
"""
)