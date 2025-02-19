import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page config for a modern wide layout
st.set_page_config(page_title="DrugX Market Share & Access Quadrants", layout="wide")

st.markdown("## DrugX Market Share & Access Quadrants")

# Cache the data loading for performance
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Sidebar: File uploader and variable selection
st.sidebar.header("Upload Your Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    st.sidebar.header("Select Variables for Chart")
    x_var = st.sidebar.selectbox("Select X-axis Variable", data.columns, index=0)
    y_var = st.sidebar.selectbox("Select Y-axis Variable", data.columns, index=1)
    
    if x_var and y_var:
        x = data[x_var]
        y = data[y_var]
    else:
        st.sidebar.error("Please select valid variables for both axes.")
        st.stop()
else:
    # Generate dummy data if no file is uploaded
    np.random.seed(42)
    x = np.random.uniform(10, 50, 100)
    y = np.random.uniform(0.5, 3.5, 100)
    data = pd.DataFrame({"X_Value": x, "Y_Value": y})
    x_var = "X_Value"
    y_var = "Y_Value"

# Compute min, max, and default thresholds
x_min, x_max = float(x.min()), float(x.max())
y_min, y_max = float(y.min()), float(y.max())
x_threshold_default = float(x.mean())
y_threshold_default = float(y.mean())

# Sidebar: Adjust quadrant thresholds
st.sidebar.header("Adjust Quadrant Thresholds")
x_threshold = st.sidebar.slider("X-axis Threshold", min_value=x_min, max_value=x_max, value=x_threshold_default)
y_threshold = st.sidebar.slider("Y-axis Threshold", min_value=y_min, max_value=y_max, value=y_threshold_default)

# Optimized categorization using numpy.select
conditions = [
    (x >= x_threshold) & (y >= y_threshold),
    (x >= x_threshold) & (y < y_threshold),
    (x < x_threshold) & (y >= y_threshold),
    (x < x_threshold) & (y < y_threshold)
]
choices = [
    'High Marketshare & High Access', 
    'Low Marketshare & High Access', 
    'High Marketshare & Low Access', 
    'Low Marketshare & Low Access'
]
data["Category"] = np.select(conditions, choices, default="Unknown")

# Search functionality: Filter the dataframe based on query across all columns
def search_data(query, df):
    mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
    return df[mask.any(axis=1)]

search_term = st.text_input("Search Dataset")
if search_term:
    search_results = search_data(search_term, data)
    st.write("### Search Results:")
    st.dataframe(search_results)
    if not search_results.empty:
        st.write("### AI-Generated Insights:")
        st.write(f"🔍 The search term '{search_term}' appears in the dataset. Here are some potential insights:")
        if len(search_results) > 10:
            st.write("📊 The term appears frequently, indicating a strong trend in this data segment.")
        elif len(search_results) > 5:
            st.write("📈 Moderate occurrence, possibly indicating a growing trend.")
        else:
            st.write("🔎 The term appears infrequently, which could indicate an emerging opportunity or niche area.")

# Create interactive scatter plot with Plotly Express
fig = px.scatter(
    data, 
    x=x_var, 
    y=y_var, 
    color="Category", 
    title="DrugX Market Share & Access Quadrants", 
    labels={x_var: x_var, y_var: y_var},
    hover_data=[x_var, y_var, "Category"],
    template="plotly_white"
)

# Update marker style for sleek appearance
fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color='black')))

# Add quadrant lines
fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))

# Calculate ranges for annotation positioning
x_range = x_max - x_min
y_range = y_max - y_min

# Add quadrant annotations with refined alignment
fig.add_annotation(
    x=x_threshold + 0.1 * x_range, y=y_threshold + 0.1 * y_range,
    text="⬆ High Marketshare & High Access",
    showarrow=False, font=dict(size=12, color="black")
)
fig.add_annotation(
    x=x_threshold + 0.1 * x_range, y=y_threshold - 0.1 * y_range,
    text="⬇ Low Marketshare & High Access",
    showarrow=False, font=dict(size=12, color="black")
)
fig.add_annotation(
    x=x_threshold - 0.1 * x_range, y=y_threshold + 0.1 * y_range,
    text="⬆ High Marketshare & Low Access",
    showarrow=False, font=dict(size=12, color="black")
)
fig.add_annotation(
    x=x_threshold - 0.1 * x_range, y=y_threshold - 0.1 * y_range,
    text="⬇ Low Marketshare & Low Access",
    showarrow=False, font=dict(size=12, color="black")
)

st.plotly_chart(fig, use_container_width=True)

# Sidebar: Download data button
st.sidebar.header("Download Data")
csv = data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download Data as CSV", csv, "marketshare_data.csv", "text/csv")

# Display key insights with modern styling
st.subheader("Key Insights")
st.markdown("✅ **High market share but low access** indicates potential for improved access strategies.")
st.markdown("✅ **Quadrant segmentation** helps pinpoint regions requiring targeted efforts for growth.")
st.markdown("✅ **Dynamic controls** allow real-time data-driven strategic analysis.")

