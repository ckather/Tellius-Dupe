import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(page_title="Quadrant Analysis with Semantic Search", layout="wide")
st.title("Quadrant Analysis with Semantic Search")

# ----------------------------------------
# 2. DATA LOADING FUNCTION
# ----------------------------------------
@st.cache_data
def load_data(file):
    """
    Load CSV or Excel file into a DataFrame.
    If file type is not supported, returns an empty DataFrame.
    """
    file_name = file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return pd.DataFrame()

# ----------------------------------------
# 3. CREATE TEXT REPRESENTATION OF EACH ROW
# ----------------------------------------
@st.cache_data
def create_row_texts(df):
    """Concatenate all columns of each row into a single string."""
    return df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()

# ----------------------------------------
# 4. LOAD THE NLP MODEL (Sentence Transformers)
# ----------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    # Using the lightweight model 'all-MiniLM-L6-v2'
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------------------
# 5. GENERATE EMBEDDINGS FOR THE DATASET
# ----------------------------------------
@st.cache_data
def get_embeddings(texts):
    """Generate embeddings for a list of texts using the Sentence Transformer model."""
    return model.encode(texts)

# ----------------------------------------
# 6. SIDEBAR: UPLOAD FILE
# ----------------------------------------
st.sidebar.header("Upload Your Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data.empty:
        st.stop()
    # Once a file is uploaded, create texts & embeddings:
    row_texts = create_row_texts(data)
    with st.spinner("Generating embeddings for your dataset..."):
        dataset_embeddings = get_embeddings(row_texts)
    st.sidebar.success("Dataset loaded and embeddings generated!")
else:
    st.info("Please upload a CSV or Excel file to populate the chart.")
    data = None  # No data available

# ----------------------------------------
# 7. SEMANTIC SEARCH SECTION
# ----------------------------------------
if data is not None:
    st.subheader("Semantic Search")
    query = st.text_input("Enter a query to search semantically (e.g. 'high market share'):")

    if query:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()

        # Show top 10 matches
        top_n = 10
        top_indices = np.argsort(similarities)[::-1][:top_n]
        semantic_results = data.iloc[top_indices].copy()
        semantic_results["Similarity"] = similarities[top_indices]
        st.markdown("**Top 10 Semantic Search Results:**")
        st.dataframe(semantic_results)

# ----------------------------------------
# 8. QUADRANT VISUALIZATION
# ----------------------------------------
if data is not None:
    st.subheader("Quadrant Visualization")
    # Let the user choose columns for the quadrant plot
    col_options = list(data.columns)
    x_col = st.selectbox("Select X-axis column", col_options, index=col_options.index("Market Share") if "Market Share" in col_options else 0)
    y_col = st.selectbox("Select Y-axis column", col_options, index=col_options.index("Access Score") if "Access Score" in col_options else 1)

    # Calculate default thresholds as the mean of the selected columns
    try:
        x_mean = data[x_col].astype(float).mean()
        y_mean = data[y_col].astype(float).mean()
    except Exception as e:
        st.error("Error converting columns to numeric values for threshold calculations.")
        st.stop()

    # Allow threshold adjustment
    x_min, x_max = float(data[x_col].astype(float).min()), float(data[x_col].astype(float).max())
    y_min, y_max = float(data[y_col].astype(float).min()), float(data[y_col].astype(float).max())
    st.write("Adjust quadrant thresholds (optional):")
    x_threshold = st.slider("X Threshold", min_value=x_min, max_value=x_max, value=float(x_mean))
    y_threshold = st.slider("Y Threshold", min_value=y_min, max_value=y_max, value=float(y_mean))

    # Assign quadrant categories
    x = data[x_col].astype(float)
    y = data[y_col].astype(float)
    conditions = [
        (x >= x_threshold) & (y >= y_threshold),
        (x >= x_threshold) & (y < y_threshold),
        (x < x_threshold) & (y >= y_threshold),
        (x < x_threshold) & (y < y_threshold)
    ]
    categories = [
        'High Marketshare & High Access', 
        'Low Marketshare & High Access', 
        'High Marketshare & Low Access', 
        'Low Marketshare & Low Access'
    ]
    data["Category"] = np.select(conditions, categories, default="Unknown")

    # Create quadrant scatter plot
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="Category",
        title="Quadrant Visualization",
        template="plotly_white",
        hover_data=data.columns
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="black")))
    fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
    fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))
    st.plotly_chart(fig, use_container_width=True)

    # Optional download button for processed data
    st.sidebar.header("Download Processed Data")
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download CSV", csv_data, "processed_data.csv", "text/csv")

