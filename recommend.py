def get_top_n_recommendations(user_id, user_based_model, ratings_matrix, n=20):
    """
    Returns top-N product recommendations for a user using a trained collaborative filtering model.

    Args:
        user_id (str): The username to generate recommendations for.
        user_based_model (surprise.AlgoBase): Trained Surprise model (e.g., KNNBasic).
        ratings_matrix (pd.DataFrame): Pivot table of users vs. product ratings.
        n (int): Number of top recommendations to return.

    Returns:
        List of tuples: [(product_name, predicted_rating), ...]
    """
    if user_id not in ratings_matrix.index:
        raise ValueError(f"User ID '{user_id}' not found in the ratings matrix.")

    # Products the user already rated
    user_rated_products = ratings_matrix.loc[user_id].dropna().index
    all_products = ratings_matrix.columns
    products_to_predict = [item for item in all_products if item not in user_rated_products]

    predicted_ratings = {}
    for item in products_to_predict:
        try:
            pred = user_based_model.predict(user_id, item)
            predicted_ratings[item] = pred.est
        except:
            continue

    recommended = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:n]
    return recommended
