import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------
# 1. STREAMLIT PAGE CONFIG
# ------------------------------------------
st.set_page_config(page_title="Market Share & Access Quadrants", layout="wide")
st.title("DrugX Market Share & Access Quadrants")

# ------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------
@st.cache_data
def load_data(file):
    """Load CSV or Excel data into a Pandas DataFrame."""
    file_name = file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return pd.DataFrame()

def categorize_quadrants(df, x_col, y_col, x_thresh, y_thresh):
    """Assign quadrant categories to each row based on thresholds."""
    x = df[x_col]
    y = df[y_col]

    conditions = [
        (x >= x_thresh) & (y >= y_thresh),
        (x >= x_thresh) & (y < y_thresh),
        (x < x_thresh) & (y >= y_thresh),
        (x < x_thresh) & (y < y_thresh)
    ]
    categories = [
        'High Marketshare & High Access', 
        'Low Marketshare & High Access', 
        'High Marketshare & Low Access', 
        'Low Marketshare & Low Access'
    ]
    df["Category"] = np.select(conditions, categories, default="Unknown")
    return df

def search_data(query, df):
    """
    Substring-based search across all columns.
    Returns rows where any cell contains 'query' (case-insensitive).
    """
    if not query:
        return df
    # Convert all columns to string and search
    mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
    return df[mask.any(axis=1)]

# ------------------------------------------
# 3. SIDEBAR - DATA UPLOAD
# ------------------------------------------
st.sidebar.header("Upload Your Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Load or generate data
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data.empty:
        st.stop()  # No valid data
else:
    # Generate dummy data if no file is uploaded
    np.random.seed(42)
    dummy_x = np.random.uniform(10, 50, 100)
    dummy_y = np.random.uniform(0.5, 3.5, 100)
    data = pd.DataFrame({"X_Value": dummy_x, "Y_Value": dummy_y})

# ------------------------------------------
# 4. SIMPLE SEARCH
# ------------------------------------------
st.subheader("Search Dataset (Substring-Based)")
query = st.text_input("Type a keyword or value to filter rows:")

filtered_data = search_data(query, data)
if query:
    st.write(f"**Search Results for '{query}':**")
    if filtered_data.empty:
        st.warning("No matching results found. Try a different search term.")
    else:
        st.success(f"Found {len(filtered_data)} matching rows.")
        st.dataframe(filtered_data)

# ------------------------------------------
# 5. QUADRANT VISUALIZATION
# ------------------------------------------
st.subheader("Quadrant Visualization")

if len(data.columns) < 2:
    st.warning("Not enough columns to create a quadrant plot. Please upload a dataset with at least two columns.")
else:
    # Let user pick columns for X and Y
    col_options = list(data.columns)
    x_col = st.selectbox("X-axis column", col_options, index=0)
    y_col = st.selectbox("Y-axis column", col_options, index=1)

    # Compute default thresholds (mean)
    x_mean = data[x_col].astype(float).mean()
    y_mean = data[y_col].astype(float).mean()

    # Compute min & max for slider
    x_min, x_max = float(data[x_col].astype(float).min()), float(data[x_col].astype(float).max())
    y_min, y_max = float(data[y_col].astype(float).min()), float(data[y_col].astype(float).max())

    st.write("Adjust quadrant thresholds (optional):")
    x_threshold = st.slider("X Threshold", min_value=x_min, max_value=x_max, value=float(x_mean))
    y_threshold = st.slider("Y Threshold", min_value=y_min, max_value=y_max, value=float(y_mean))

    # Assign categories
    data = categorize_quadrants(data, x_col, y_col, x_threshold, y_threshold)

    # Create plot
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="Category",
        title="DrugX Market Share & Access Quadrants",
        hover_data=data.columns,
        template="plotly_white"
    )
    # Style markers
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color='black')))

    # Add quadrant lines
    fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
    fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 6. DOWNLOAD BUTTON
# ------------------------------------------
st.sidebar.header("Download Data")
csv_data = data.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download Data as CSV", csv_data, "marketshare_data.csv", "text/csv")

# ------------------------------------------
# 7. KEY INSIGHTS
# ------------------------------------------
st.subheader("Key Insights")
st.markdown("""
- **Quadrant analysis** helps identify which segments have high vs. low market share and access.
- **Substring-based search** quickly filters rows containing your search term.
- **Adjustable thresholds** let you experiment with different definitions of "high" vs. "low."
""")
