import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Generate dummy data
np.random.seed(42)
x = np.random.uniform(10, 50, 100)  # Avg Open PA Territory (x-axis)
y = np.random.uniform(0.5, 3.5, 100)  # Market Share Claims for Sotyktu (y-axis)

# Define default quadrant thresholds
x_threshold_default = 20  # National Avg Access
y_threshold_default = 1.75  # National Market Share Claims

# Categorize data points based on quadrants
def categorize_data(x, y, x_threshold, y_threshold):
    categories = []
    for i in range(len(x)):
        if x[i] >= x_threshold and y[i] >= y_threshold:
            categories.append('Marketshare ↑ Access ↑')
        elif x[i] >= x_threshold and y[i] < y_threshold:
            categories.append('Marketshare ↓ Access ↑')
        elif x[i] < x_threshold and y[i] >= y_threshold:
            categories.append('Marketshare ↑ Access ↓')
        else:
            categories.append('Marketshare ↓ Access ↓')
    return categories

# Streamlit UI
st.title("Marketshare for Sotyktu vs Access Quadrants")

# Sidebar for user input
st.sidebar.header("Adjust Quadrant Thresholds")
x_threshold = st.sidebar.slider("National Avg Access", min_value=10, max_value=50, value=x_threshold_default)
y_threshold = st.sidebar.slider("National Market Share Claims", min_value=0.5, max_value=3.5, value=y_threshold_default)

# Update categories based on user input
categories = categorize_data(x, y, x_threshold, y_threshold)

# Create DataFrame
data = pd.DataFrame({
    "Avg_Open_PA_Terr": x,
    "Market_Share_Claims": y,
    "Category": categories
})

# Create interactive scatter plot
fig = px.scatter(data, x="Avg_Open_PA_Terr", y="Market_Share_Claims", color="Category", 
                 title="Marketshare for Sotyktu vs Access Quadrants", 
                 labels={"Avg_Open_PA_Terr": "Avg Open PA Territory", "Market_Share_Claims": "Market Share Claims for Sotyktu"},
                 hover_data=["Avg_Open_PA_Terr", "Market_Share_Claims", "Category"])

# Add quadrant lines
fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dash"))
fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dash"))

# Show figure
st.plotly_chart(fig)

# Download button for dataset
st.sidebar.header("Download Data")
st.sidebar.download_button("Download Data as CSV", data.to_csv(index=False).encode('utf-8'), "marketshare_data.csv", "text/csv")
