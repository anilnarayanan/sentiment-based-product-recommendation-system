import streamlit as st
import pickle
import gzip
import pandas as pd
import numpy as np

# Helper functions
def load_compressed_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_models():
    model = load_compressed_pickle('models/user_cf_model.pkl.gz')
    ratings = load_compressed_pickle('models/ratings_matrix.pkl.gz')
    return model, ratings

def recommend(username, model, ratings, top_n=5):
    if username not in ratings.index:
        return []

    try:
        # Convert external user ID (username) to internal surprise ID
        inner_id = model.trainset.to_inner_uid(username)
    except ValueError:
        return []

    # Get the 10 nearest neighbors (can tune this number)
    neighbor_ids = model.get_neighbors(inner_id, k=10)
    neighbor_usernames = [model.trainset.to_raw_uid(i) for i in neighbor_ids]

    # Calculate mean ratings from neighbors
    neighbor_ratings = ratings.loc[neighbor_usernames]
    mean_scores = neighbor_ratings.mean().sort_values(ascending=False)

    # Remove items the user has already rated
    user_rated = ratings.loc[username][ratings.loc[username] > 0].index
    recommendations = mean_scores.drop(user_rated, errors='ignore')

    return recommendations.head(top_n).index.tolist()


# Streamlit UI
st.title("📦 Product Recommender")
st.write("Enter your username and get top 5 recommended products based on similar users.")

# Load models
model, ratings = load_models()

username = st.text_input("Enter username:")
if st.button("Get Recommendations"):
    if username.strip() == "":
        st.warning("Please enter a valid username.")
    else:
        top_items = recommend(username, model, ratings)
        if top_items:
            st.success("Top 5 Recommendations:")
            for i, item in enumerate(top_items, 1):
                st.write(f"{i}. {item}")
        else:
            st.error("Username not found or no recommendations available.")
