import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit UI
st.title("Marketshare for Sotyktu vs Access Quadrants")

# File uploader for user-provided dataset
st.sidebar.header("Upload Your Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load user data
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # Ensure required columns exist
    required_columns = ["Avg_Open_PA_Terr", "Market_Share_Claims"]
    if all(col in data.columns for col in required_columns):
        x = data["Avg_Open_PA_Terr"]
        y = data["Market_Share_Claims"]
    else:
        st.sidebar.error("Uploaded file must contain 'Avg_Open_PA_Terr' and 'Market_Share_Claims' columns.")
        st.stop()
else:
    # Generate dummy data if no file is uploaded
    np.random.seed(42)
    x = np.random.uniform(10, 50, 100)  # Avg Open PA Territory (x-axis)
    y = np.random.uniform(0.5, 3.5, 100)  # Market Share Claims for Sotyktu (y-axis)
    data = pd.DataFrame({"Avg_Open_PA_Terr": x, "Market_Share_Claims": y})

# Define default quadrant thresholds
x_threshold_default = 20  # National Avg Access
y_threshold_default = 1.75  # National Market Share Claims

# Sidebar for user input
st.sidebar.header("Adjust Quadrant Thresholds")
x_threshold = st.sidebar.slider("National Avg Access", min_value=int(min(x)), max_value=int(max(x)), value=x_threshold_default)
y_threshold = st.sidebar.slider("National Market Share Claims", min_value=float(min(y)), max_value=float(max(y)), value=y_threshold_default)

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

# Update categories based on user input
categories = categorize_data(x, y, x_threshold, y_threshold)
data["Category"] = categories

# Create sleek interactive scatter plot
fig = px.scatter(data, x="Avg_Open_PA_Terr", y="Market_Share_Claims", color="Category", 
                 title="Marketshare for Sotyktu vs Access Quadrants", 
                 labels={"Avg_Open_PA_Terr": "Avg Open PA Territory", "Market_Share_Claims": "Market Share Claims for Sotyktu"},
                 hover_data=["Avg_Open_PA_Terr", "Market_Share_Claims", "Category"],
                 template="plotly_white")

# Remove bubble-like appearance, set minimal sleek markers
fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color='black')))

# Add quadrant lines with sleek style
fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))

# Refine quadrant annotations with better alignment
fig.add_annotation(x=x_threshold + 10, y=y_threshold + 0.5,
                   text="⬆ High Marketshare & High Access",
                   showarrow=False, font=dict(size=12, color="black"))
fig.add_annotation(x=x_threshold + 10, y=y_threshold - 0.5,
                   text="⬇ Low Marketshare & High Access",
                   showarrow=False, font=dict(size=12, color="black"))
fig.add_annotation(x=x_threshold - 10, y=y_threshold + 0.5,
                   text="⬆ High Marketshare & Low Access",
                   showarrow=False, font=dict(size=12, color="black"))
fig.add_annotation(x=x_threshold - 10, y=y_threshold - 0.5,
                   text="⬇ Low Marketshare & Low Access",
                   showarrow=False, font=dict(size=12, color="black"))

# Show figure
st.plotly_chart(fig)

# Download button for dataset
st.sidebar.header("Download Data")
st.sidebar.download_button("Download Data as CSV", data.to_csv(index=False).encode('utf-8'), "marketshare_data.csv", "text/csv")

# Display key insights with modern styling
st.subheader("Key Insights")
st.markdown("✅ **High market share but low access** indicates potential for improved access strategies.")
st.markdown("✅ **Quadrant segmentation** helps pinpoint regions requiring targeted efforts for growth.")
st.markdown("✅ **Dynamic controls** allow real-time data-driven strategic analysis.")
