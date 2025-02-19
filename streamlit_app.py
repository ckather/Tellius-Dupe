import streamlit as st
import sys
import subprocess

# -------------------------------------------------------------------
# 1) FORCE-INSTALL SPECIFIC VERSIONS (PINNED OPENAI==0.28.0)
# -------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "openai==0.28.0",      # Pin to old version that supports openai.Embedding.create()
    "pandas",
    "numpy",
    "plotly",
    "scikit-learn",
    "tiktoken"
]

for pkg in REQUIRED_PACKAGES:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", pkg])

# Now import after forced install
import openai
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 2) OPENAI API KEY CONFIG
# -------------------------------------------------------------------
# Option A: Hard-code your key (NOT recommended)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Option B: Use Streamlit Secrets (recommended):
# 1. In Streamlit Cloud, go to "Manage app" -> "Secrets".
# 2. Add a secret named OPENAI_API_KEY.
# 3. Uncomment this line:
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------------------------------------------------
# 3) STREAMLIT APP SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="NLP Quadrant App (Pinned openai==0.28.0)", layout="wide")
st.title("DrugX Market Share & Access Quadrants (NLP Search)")

@st.cache_data
def load_data(file):
    """Load CSV or Excel data into a Pandas DataFrame."""
    file_name = file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel.")
        return pd.DataFrame()

@st.cache_data
def get_openai_embeddings(texts, model_name="text-embedding-ada-002"):
    """
    Create embeddings for a list of strings using openai==0.28.0 style.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    # For large lists, you may want to batch requests
    response = openai.Embedding.create(input=texts, model=model_name)
    embeddings = [item["embedding"] for item in response["data"]]
    return np.array(embeddings)

def generate_quadrant_categories(df, x_col, y_col, x_threshold, y_threshold):
    """
    Add a 'Category' column to df based on quadrant thresholds.
    """
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    conditions = [
        (x >= x_threshold) & (y >= y_threshold),
        (x >= x_threshold) & (y < y_threshold),
        (x < x_threshold) & (y >= y_threshold),
        (x < x_threshold) & (y < y_threshold)
    ]
    choices = [
        "High Marketshare & High Access", 
        "Low Marketshare & High Access", 
        "High Marketshare & Low Access", 
        "Low Marketshare & Low Access"
    ]
    df["Category"] = np.select(conditions, choices, default="Unknown")
    return df

# -------------------------------------------------------------------
# 4) SIDEBAR & DATA LOADING
# -------------------------------------------------------------------
st.sidebar.title("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data.empty:
        st.stop()
else:
    # Fallback: generate dummy data if no file is uploaded
    np.random.seed(42)
    dummy_x = np.random.uniform(10, 50, 100)
    dummy_y = np.random.uniform(0.5, 3.5, 100)
    data = pd.DataFrame({"X_Value": dummy_x, "Y_Value": dummy_y})

# -------------------------------------------------------------------
# 5) EMBEDDINGS GENERATION
# -------------------------------------------------------------------
st.sidebar.write("Generating embeddings... (one-time, pinned openai==0.28.0)")
string_data = data.astype(str)
row_texts = string_data.apply(lambda row: " | ".join(row.values), axis=1).tolist()

dataset_embeddings = get_openai_embeddings(row_texts)
st.sidebar.success("Embeddings generated!")

# -------------------------------------------------------------------
# 6) NLP-BASED SEARCH
# -------------------------------------------------------------------
st.subheader("NLP-Based Search")

user_query = st.text_input("Ask a question or type a keyword:")
if user_query:
    query_embedding = get_openai_embeddings([user_query])
    similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()

    # Sort by similarity descending
    top_n = 10
    top_indices = np.argsort(similarities)[::-1][:top_n]
    search_results = data.iloc[top_indices].copy()
    search_results["Similarity"] = similarities[top_indices]

    st.markdown(f"**Top {top_n} matches** for your query:")
    st.dataframe(search_results)

    avg_score = np.mean(similarities[top_indices])
    if avg_score > 0.85:
        st.info("AI Insight: Very strong match. The dataset strongly contains your query.")
    elif avg_score > 0.65:
        st.info("AI Insight: Moderate match. The concept is somewhat present.")
    else:
        st.info("AI Insight: Weak match. The dataset doesn't strongly contain your query.")

# -------------------------------------------------------------------
# 7) QUADRANT LOGIC & PLOT
# -------------------------------------------------------------------
st.subheader("Quadrant Visualization")

if len(data.columns) >= 2:
    col_options = list(data.columns)
    x_col = st.selectbox("X-axis column", col_options, index=0)
    y_col = st.selectbox("Y-axis column", col_options, index=1)

    # Threshold defaults
    x_mean = data[x_col].astype(float).mean()
    y_mean = data[y_col].astype(float).mean()

    x_min, x_max = data[x_col].astype(float).min(), data[x_col].astype(float).max()
    y_min, y_max = data[y_col].astype(float).min(), data[y_col].astype(float).max()

    st.write("Adjust quadrant thresholds (optional):")
    x_threshold = st.slider("X Threshold", min_value=float(x_min), max_value=float(x_max), value=float(x_mean))
    y_threshold = st.slider("Y Threshold", min_value=float(y_min), max_value=float(y_max), value=float(y_mean))

    data = generate_quadrant_categories(data, x_col, y_col, x_threshold, y_threshold)

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="Category",
        title="DrugX Market Share & Access Quadrants",
        hover_data=data.columns,
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="black")))
    fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
    fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough columns to plot quadrants. Please upload a dataset with at least two columns.")

# -------------------------------------------------------------------
# 8) DOWNLOAD BUTTON
# -------------------------------------------------------------------
st.sidebar.subheader("Download Processed Data")
csv_data = data.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download CSV", csv_data, "nlp_quadrant_data.csv", "text/csv")

# -------------------------------------------------------------------
# 9) WRAP-UP INSIGHTS
# -------------------------------------------------------------------
st.write("### Key Insights")
st.markdown("""
- **Pinned openai==0.28.0** so `openai.Embedding.create()` works without the APIRemovedInV1 error.
- **NLP-based search** finds rows related to your query by semantic meaning.
- **Quadrant analysis** helps identify high vs. low market share and access segments.
- **Adjust thresholds** to see real-time changes in quadrant classification.
""")

st.info("If you still see APIRemovedInV1, your environment might have forcibly installed openai>=1.0.0. Try a fresh environment or remove the newer package first.")
