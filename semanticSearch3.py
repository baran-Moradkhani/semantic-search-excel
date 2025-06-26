#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import io
import hashlib

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- UI SECTION ---
st.title("🧠 Semantic Search in Excel")

uploaded_file = st.file_uploader("📁 Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("✅ Data Preview", df.head())

    selected_columns = st.multiselect("📌 Select columns to include in semantic search", df.columns.tolist())

    if selected_columns:
        df["searchable_text"] = df[selected_columns].astype(str).agg(" ".join, axis=1)

        # ✅ Create a unique hash for selected columns
        cols_string = "_".join(selected_columns)
        cols_hash = hashlib.md5(cols_string.encode()).hexdigest()

        # ✅ Update cache filename to include both file name and column hash
        embedding_cache_file = f"{uploaded_file.name}_{cols_hash}_embeddings.pkl"

        embeddings = None  # ✅ Initialize to None

        # ✅ Always check and load cache FIRST (even before button)
        if os.path.exists(embedding_cache_file):
            st.info("📦 Loading cached embeddings...")
            with open(embedding_cache_file, "rb") as f:
                embeddings = pickle.load(f)
            df["embedding"] = embeddings

        # ✅ Only generate embeddings if button clicked AND no cache
        elif st.button("🧬 Generate Embeddings"):
            with st.spinner("Generating embeddings..."):
                texts = df["searchable_text"].tolist()
                try:
                    response = openai.Embedding.create(
                        model="text-embedding-3-small",
                        input=texts
                    )
                    embeddings = [item["embedding"] for item in response["data"]]
                    df["embedding"] = embeddings

                    with open(embedding_cache_file, "wb") as f:
                        pickle.dump(embeddings, f)

                    st.success("✅ Embeddings generated and cached.")
                except Exception as e:
                    st.error(f"Embedding error: {e}")
                    df["embedding"] = [[0.0] * 1536] * len(df)

        query = st.text_input("🔍 Enter a semantic search query")
        top_n = st.slider("Top N results", 1, 20, 5)

        # ✅ Extra check for embeddings
        if query and "embedding" in df.columns and embeddings is not None:
            with st.spinner("Searching..."):

                def get_embedding(text):
                    try:
                        response = openai.Embedding.create(
                            model="text-embedding-3-small",
                            input=text
                        )
                        return response['data'][0]['embedding']
                    except Exception as e:
                        st.error(f"Query embedding error: {e}")
                        return [0.0] * 1536

                query_embedding = np.array(get_embedding(query)).reshape(1, -1)
                matrix = np.vstack(df["embedding"].values)

                df["similarity"] = cosine_similarity(query_embedding, matrix)[0]
                results = df.sort_values(by="similarity", ascending=False).head(top_n)

                st.write("🔎 Top Results", results[selected_columns + ["similarity"]])

                output = io.BytesIO()
                results.to_excel(output, index=False)
                output.seek(0)
                st.download_button(
                    "💾 Download Results as Excel",
                    data=output,
                    file_name="search_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


# In[ ]:




