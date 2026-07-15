import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# Load Model
model = joblib.load("sales_forecasting_model.pkl")

# Load Dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Title
st.title("📈 Sales Forecasting using Machine Learning")
st.write("Predict future sales using a Linear Regression model.")

st.divider()
# Sidebar
st.sidebar.title("Sales Forecasting")
st.sidebar.write("Enter Year and Month to predict sales.")

# User Input
year = st.sidebar.number_input(
    "Enter Year",
    min_value=2018,
    max_value=2035,
    value=2018
)

month = st.sidebar.selectbox(
    "Select Month",
    [1,2,3,4,5,6,7,8,9,10,11,12]
)

# Predict Button
if st.sidebar.button("Predict Sales"):

    future = pd.DataFrame({
        "Year": [year],
        "Month": [month]
    })

    prediction = model.predict(future)

    st.success(f"Predicted Sales: ₹ {prediction[0]:,.2f}")
    st.divider()

st.header("📋 Dataset Overview")

st.write("First 5 Rows")

st.dataframe(df.head())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Sales", f"₹ {df['Sales'].sum():,.0f}")

with col3:
    st.metric("Average Sales", f"₹ {df['Sales'].mean():,.2f}")
    st.divider()

st.header("📈 Monthly Sales Trend")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create Month column
df["Month"] = df["Order Date"].dt.month

# Monthly Sales
monthly_sales = df.groupby("Month")["Sales"].sum()

# Plot
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

ax.set_title("Monthly Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.grid(True)

st.pyplot(fig)
st.divider()

st.header("📦 Sales by Category")

category_sales = df.groupby("Category")["Sales"].sum()

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    category_sales.index,
    category_sales.values
)

ax.set_title("Sales by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Sales")

st.pyplot(fig)
st.divider()

st.header("🌍 Sales by Region")

region_sales = df.groupby("Region")["Sales"].sum()

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    region_sales.index,
    region_sales.values
)

ax.set_title("Sales by Region")
ax.set_xlabel("Region")
ax.set_ylabel("Sales")

st.pyplot(fig)
st.divider()

st.header("📅 Year-wise Sales")

df["Year"] = df["Order Date"].dt.year

year_sales = df.groupby("Year")["Sales"].sum()

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    year_sales.index,
    year_sales.values,
    marker="o",
    linewidth=2
)

ax.set_title("Year-wise Sales")
ax.set_xlabel("Year")
ax.set_ylabel("Sales")
ax.grid(True)

st.pyplot(fig)
st.divider()

st.header("📌 About This Project")

st.write("""
This Sales Forecasting project uses a **Linear Regression** machine learning model to predict future sales based on **Year** and **Month**.

### Workflow
- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Sales Prediction
- Streamlit Deployment
""")