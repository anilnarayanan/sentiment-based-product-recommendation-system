import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings matrix only
def load_compressed_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_ratings():
    return load_compressed_pickle('models/ratings_matrix.pkl.gz')

def recommend(username, ratings, top_n=5, neighbor_k=10):
    if username not in ratings.index:
        return []

    similarity_matrix = pd.DataFrame(
        cosine_similarity(ratings.fillna(0)),
        index=ratings.index,
        columns=ratings.index
    )

    neighbors = similarity_matrix.loc[username].drop(username)
    top_neighbors = neighbors.sort_values(ascending=False).head(neighbor_k).index

    neighbor_ratings = ratings.loc[top_neighbors]
    mean_scores = neighbor_ratings.mean().sort_values(ascending=False)

    user_rated = ratings.loc[username][ratings.loc[username] > 0].index
    recommendations = mean_scores.drop(user_rated, errors='ignore')

    return recommendations.head(top_n).index.tolist()

# Streamlit UI
st.title("📦 Product Recommender")
st.write("Enter your username and get top 5 recommended products based on similar users.")

ratings = load_ratings()

username = st.text_input("Enter username:")
if st.button("Get Recommendations"):
    if username.strip() == "":
        st.warning("Please enter a valid username.")
    else:
        top_items = recommend(username, ratings)
        if top_items:
            st.success("Top 5 Recommendations:")
            for i, item in enumerate(top_items, 1):
                st.write(f"{i}. {item}")
        else:
            st.error("Username not found or no recommendations available.")
