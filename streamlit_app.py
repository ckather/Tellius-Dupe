import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Semantic Search Demo", layout="wide")
st.title("Semantic Search on Your Dataset")

# -----------------------------
# 2. DATA LOADING
# -----------------------------
@st.cache_data
def load_data(file):
    """Load CSV or Excel file into a DataFrame."""
    file_name = file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload CSV or Excel.")
        return pd.DataFrame()

# Let user upload a file, or use dummy data
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    data = load_data(uploaded_file)
    if data.empty:
        st.stop()
else:
    st.warning("No file uploaded; using dummy data.")
    np.random.seed(42)
    dummy_x = np.random.uniform(10, 50, 100)
    dummy_y = np.random.uniform(0.5, 3.5, 100)
    data = pd.DataFrame({"X_Value": dummy_x, "Y_Value": dummy_y})

# -----------------------------
# 3. CREATE TEXT REPRESENTATION FOR EACH ROW
# -----------------------------
@st.cache_data
def create_row_texts(df):
    """Convert each row to a single string for semantic search."""
    # You can customize this to include only the columns you care about
    return df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()

row_texts = create_row_texts(data)

# -----------------------------
# 4. LOAD THE NLP MODEL (Sentence Transformers)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    # This loads a lightweight model (all-MiniLM-L6-v2) for semantic search.
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# 5. GENERATE EMBEDDINGS FOR THE DATASET
# -----------------------------
@st.cache_data
def get_embeddings(texts):
    """Generate embeddings for a list of texts using Sentence Transformers."""
    return model.encode(texts)

with st.spinner("Generating embeddings for your dataset..."):
    dataset_embeddings = get_embeddings(row_texts)
st.success("Dataset embeddings generated!")

# -----------------------------
# 6. SEMANTIC SEARCH FUNCTIONALITY
# -----------------------------
st.subheader("Semantic Search")
query = st.text_input("Enter a query to search semantically:")

if query:
    # Compute the embedding for the query
    query_embedding = model.encode([query])
    # Compute cosine similarity between query and each row
    similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()
    
    # Get top 10 matches
    top_n = 10
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Prepare results
    semantic_results = data.iloc[top_indices].copy()
    semantic_results["Similarity"] = similarities[top_indices]
    
    st.markdown("**Top 10 Semantic Search Results:**")
    st.dataframe(semantic_results)

# -----------------------------
# 7. OPTIONAL: QUADRANT VISUALIZATION (from your earlier code)
# -----------------------------
st.subheader("Quadrant Visualization")
if len(data.columns) >= 2:
    col_options = list(data.columns)
    x_col = st.selectbox("Select X-axis column", col_options, index=0)
    y_col = st.selectbox("Select Y-axis column", col_options, index=1)

    # Compute default thresholds (mean)
    x_mean = data[x_col].astype(float).mean()
    y_mean = data[y_col].astype(float).mean()

    # Slider for thresholds
    x_min, x_max = float(data[x_col].astype(float).min()), float(data[x_col].astype(float).max())
    y_min, y_max = float(data[y_col].astype(float).min()), float(data[y_col].astype(float).max())
    st.write("Adjust quadrant thresholds (optional):")
    x_threshold = st.slider("X Threshold", min_value=x_min, max_value=x_max, value=float(x_mean))
    y_threshold = st.slider("Y Threshold", min_value=y_min, max_value=y_max, value=float(y_mean))

    # Categorize data points based on quadrants
    conditions = [
        (data[x_col].astype(float) >= x_threshold) & (data[y_col].astype(float) >= y_threshold),
        (data[x_col].astype(float) >= x_threshold) & (data[y_col].astype(float) < y_threshold),
        (data[x_col].astype(float) < x_threshold) & (data[y_col].astype(float) >= y_threshold),
        (data[x_col].astype(float) < x_threshold) & (data[y_col].astype(float) < y_threshold)
    ]
    categories = [
        'High Marketshare & High Access',
        'Low Marketshare & High Access',
        'High Marketshare & Low Access',
        'Low Marketshare & Low Access'
    ]
    data["Category"] = np.select(conditions, categories, default="Unknown")

    # Create scatter plot
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="Category",
        title="Quadrant Visualization",
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color='black')))
    fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
    fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough columns to display quadrant visualization.")

# -----------------------------
# 8. DOWNLOAD BUTTON (Optional)
# -----------------------------
st.sidebar.header("Download Data")
csv_data = data.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download Data as CSV", csv_data, "data.csv", "text/csv")

# -----------------------------
# 9. NOTES
# -----------------------------
st.markdown("""
**How It Works:**  
- The app loads your dataset (or uses dummy data if none is uploaded).  
- Each row is turned into a single text string (combining all column values).  
- The Sentence Transformers model generates a numerical 'embedding' for each row.  
- When you enter a query, the app generates an embedding for your query and computes similarity scores between your query and each row.  
- The top matching rows (based on similarity) are displayed.
""")

