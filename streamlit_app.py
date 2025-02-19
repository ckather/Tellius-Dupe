import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit UI
st.markdown("## DrugX Market Share & Access Quadrants")

# File uploader for user-provided dataset
st.sidebar.header("Upload Your Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load user data
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # Allow user to select variables for x and y axes
    st.sidebar.header("Select Variables for Chart")
    x_var = st.sidebar.selectbox("Select X-axis Variable", data.columns, index=0)
    y_var = st.sidebar.selectbox("Select Y-axis Variable", data.columns, index=1)
    
    # Ensure valid selections
    if x_var and y_var:
        x = data[x_var]
        y = data[y_var]
    else:
        st.sidebar.error("Please select valid variables for both axes.")
        st.stop()
else:
    # Generate dummy data if no file is uploaded
    np.random.seed(42)
    x = np.random.uniform(10, 50, 100)  # Generic X-axis values
    y = np.random.uniform(0.5, 3.5, 100)  # Generic Y-axis values
    data = pd.DataFrame({"X_Value": x, "Y_Value": y})
    x_var = "X_Value"
    y_var = "Y_Value"

# Define default quadrant thresholds
x_threshold_default = x.mean()  # Dynamic default threshold based on data
y_threshold_default = y.mean()

# Searchability feature
def search_data(query, dataframe):
    filtered_df = dataframe[dataframe.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    return filtered_df

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

# Sidebar for user input
st.sidebar.header("Adjust Quadrant Thresholds")
x_threshold = st.sidebar.slider("X-axis Threshold", min_value=float(min(x)), max_value=float(max(x)), value=float(x_threshold_default))
y_threshold = st.sidebar.slider("Y-axis Threshold", min_value=float(min(y)), max_value=float(max(y)), value=float(y_threshold_default))

# Categorize data points based on quadrants
def categorize_data(x, y, x_threshold, y_threshold):
    categories = []
    for i in range(len(x)):
        if x[i] >= x_threshold and y[i] >= y_threshold:
            categories.append('High Marketshare & High Access')
        elif x[i] >= x_threshold and y[i] < y_threshold:
            categories.append('Low Marketshare & High Access')
        elif x[i] < x_threshold and y[i] >= y_threshold:
            categories.append('High Marketshare & Low Access')
        else:
            categories.append('Low Marketshare & Low Access')
    return categories

# Update categories based on user input
categories = categorize_data(x, y, x_threshold, y_threshold)
data["Category"] = categories

# Create interactive scatter plot
fig = px.scatter(data, x=x_var, y=y_var, color="Category", 
                 title="DrugX Market Share & Access Quadrants", 
                 labels={x_var: x_var, y_var: y_var},
                 hover_data=[x_var, y_var, "Category"],
                 template="plotly_white")

# Remove bubble-like appearance, set minimal sleek markers
fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color='black')))

# Add quadrant lines with sleek style
fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))

# Refine quadrant annotations with better alignment
fig.add_annotation(x=x_threshold + (max(x) - min(x)) * 0.1, y=y_threshold + (max(y) - min(y)) * 0.1,
                   text="⬆ High Marketshare & High Access",
                   showarrow=False, font=dict(size=12, color="black"))
fig.add_annotation(x=x_threshold + (max(x) - min(x)) * 0.1, y=y_threshold - (max(y) - min(y)) * 0.1,
                   text="⬇ Low Marketshare & High Access",
                   showarrow=False, font=dict(size=12, color="black"))
fig.add_annotation(x=x_threshold - (max(x) - min(x)) * 0.1, y=y_threshold + (max(y) - min(y)) * 0.1,
                   text="⬆ High Marketshare & Low Access",
                   showarrow=False, font=dict(size=12, color="black"))
fig.add_annotation(x=x_threshold - (max(x) - min(x)) * 0.1, y=y_threshold - (max(y) - min(y)) * 0.1,
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
