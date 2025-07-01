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

@st.cache_data
def get_neighbors_map(ratings, k=10):
    """
    Precompute neighbors list for every user.
    Returns a dict: user -> list of neighbors.
    """
    # Fill missing with 0
    mat = ratings.fillna(0).values
    sims = cosine_similarity(mat)  # shape (n_users, n_users)
    users = ratings.index.tolist()
    neigh = {}
    for i, u in enumerate(users):
        sim_series = pd.Series(sims[i], index=users)
        # drop itself and keep top k
        top = sim_series.drop(u).nlargest(k).index.tolist()
        neigh[u] = top
    return neigh

def recommend(username, ratings, neighbors_map, top_n=5):
    if username not in ratings.index:
        return []

    neighbors = neighbors_map.get(username, [])
    if not neighbors:
        return []

    # Mean scores of neighbors
    neighbor_ratings = ratings.loc[neighbors]
    mean_scores = neighbor_ratings.mean().sort_values(ascending=False)

    # Exclude already-rated items
    user_rated = ratings.loc[username][ratings.loc[username] > 0].index
    recommendations = mean_scores.drop(user_rated, errors='ignore')

    return recommendations.head(top_n).index.tolist()

# Streamlit UI
st.title("📦 Product Recommender")
st.write("Enter your username and get top 5 recommended products based on similar users.")

ratings = load_ratings()
neighbors_map = get_neighbors_map(ratings, k=10)

username = st.text_input("Enter username:")
if st.button("Get Recommendations"):
    username = username.strip()
    if not username:
        st.warning("Please enter a valid username.")
    else:
        top_items = recommend(username, ratings, neighbors_map)
        if top_items:
            st.success("Top 5 Recommendations:")
            for i, item in enumerate(top_items, 1):
                st.write(f"{i}. {item}")
        else:
            st.error("Username not found or no recommendations available.")
