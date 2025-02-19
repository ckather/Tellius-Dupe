import streamlit as st
import pandas as pd
import numpy as np
import openai
import plotly.express as px
import tiktoken
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. SETUP
# -----------------------------
# Replace with your actual OpenAI API key (or store in st.secrets for security)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Streamlit config
st.set_page_config(page_title="NLP-Based Search Demo", layout="wide")
st.title("DrugX Market Share & Access Quadrants (NLP Search)")

# A small utility function to get embeddings from OpenAI
@st.cache_data
def get_openai_embeddings(texts: List[str], model_name: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Takes a list of text strings and returns a numpy array of shape (len(texts), embedding_dim).
    """
    # Note: For large lists, you should batch requests to avoid rate limits
    response = openai.Embedding.create(input=texts, model=model_name)
    embeddings = [r["embedding"] for r in response["data"]]
    return np.array(embeddings)

# A function to load CSV or Excel
@st.cache_data
def load_data(file):
    file_name = file.name.lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(file)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return pd.DataFrame()

# -----------------------------
# 2. DATA LOADING
# -----------------------------
st.sidebar.header("Upload Your Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data.empty:
        st.stop()
else:
    # Fallback to dummy data
    np.random.seed(42)
    x = np.random.uniform(10, 50, 100)
    y = np.random.uniform(0.5, 3.5, 100)
    data = pd.DataFrame({"X_Value": x, "Y_Value": y})

# Force data to string columns for embedding
# (In practice, you'd combine numeric fields into textual descriptions or handle them separately.)
string_data = data.astype(str)
row_texts = string_data.apply(lambda row: " | ".join(row.values), axis=1).tolist()

# -----------------------------
# 3. EMBEDDINGS
# -----------------------------
st.sidebar.write("Generating embeddings for the dataset (one-time). Please wait...")
dataset_embeddings = get_openai_embeddings(row_texts)
embedding_dim = dataset_embeddings.shape[1]
st.sidebar.success("Embeddings generated successfully!")

# -----------------------------
# 4. USER INPUT & SEMANTIC SEARCH
# -----------------------------
st.subheader("NLP-Based Search")

user_query = st.text_input("Ask a question or type a keyword:")
if user_query:
    # Get embedding for the user's query
    query_embedding = get_openai_embeddings([user_query])  # shape (1, embedding_dim)

    # Compute cosine similarities between the query and each row
    similarities = cosine_similarity(query_embedding, dataset_embeddings)  # shape (1, num_rows)
    similarities = similarities.flatten()  # shape (num_rows,)

    # Sort rows by similarity (descending)
    top_n = 10  # number of results to show
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Prepare results
    search_results = data.iloc[top_indices].copy()
    search_results["Similarity"] = similarities[top_indices]

    st.markdown(f"**Top {top_n} semantic matches** for your query:")
    st.dataframe(search_results)

    # Optional: Basic AI "insights" based on similarity scores
    avg_score = np.mean(similarities[top_indices])
    if avg_score > 0.85:
        st.write("**AI Insight**: The query is strongly represented in the dataset. High average similarity suggests a clear match.")
    elif avg_score > 0.65:
        st.write("**AI Insight**: The query partially matches multiple rows. Moderate similarity suggests the concept is somewhat present.")
    else:
        st.write("**AI Insight**: The query does not strongly match the dataset. Results may be loosely related.")

# -----------------------------
# 5. QUADRANT LOGIC (OPTIONAL)
# -----------------------------
# Example: We'll assume the first two columns are X and Y for quadrant logic
# If you have actual variable selection, replicate that from your existing code
if len(data.columns) >= 2:
    x_var, y_var = data.columns[0], data.columns[1]
    x = data[x_var]
    y = data[y_var]
    
    # Basic quadrant thresholds
    x_threshold = x.mean()
    y_threshold = y.mean()

    # Quadrant categories
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

    fig = px.scatter(
        data, 
        x=x_var, 
        y=y_var, 
        color="Category", 
        title="DrugX Market Share & Access Quadrants", 
        hover_data=data.columns,
        template="plotly_white"
    )

    # Add quadrant lines
    fig.add_vline(x=x_threshold, line=dict(color="red", width=2, dash="dot"))
    fig.add_hline(y=y_threshold, line=dict(color="red", width=2, dash="dot"))

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6. DOWNLOAD BUTTON
# -----------------------------
csv_data = data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download Data as CSV", csv_data, "marketshare_data.csv", "text/csv")
