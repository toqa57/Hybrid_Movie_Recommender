import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from collections import defaultdict


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender system using matrix factorization
    """

    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.user_factors = None
        self.movie_factors = None
        self.user_ids = None
        self.movie_ids = None
        self.mean_rating = None
        self.movies_df = None
        self.trained = False

    def fit(self, ratings_df, movies_df=None):
        # Store movie information
        self.movies_df = movies_df

        # Create user-item matrix
        self.mean_rating = ratings_df['rating'].mean()
        user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        # Store user and movie IDs
        self.user_ids = user_item_matrix.index.values
        self.movie_ids = user_item_matrix.columns.values

        # Apply Truncated SVD (similar to matrix factorization)
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = svd.fit_transform(user_item_matrix)
        self.movie_factors = svd.components_.T

        self.trained = True
        print(f"Collaborative filtering model trained with {len(ratings_df)} ratings")

    def predict_rating(self, user_id, movie_id):
        if not self.trained:
            raise Exception("Model not trained. Call fit() first.")

        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            movie_idx = np.where(self.movie_ids == movie_id)[0][0]
            pred = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
            return np.clip(pred, 1, 5)  # Clip to rating range 1-5
        except IndexError:
            return self.mean_rating  # Return average if unknown user/movie

    def get_top_n_recommendations(self, user_id, n=10, min_rating=0, exclude_rated=True, ratings_df=None):
        if not self.trained:
            raise Exception("Model not trained. Call fit() first.")

        if self.movies_df is None:
            raise Exception("Movie information not available. Provide movies_df in fit().")

        # Get all movie IDs
        all_movie_ids = self.movies_df['movieId'].unique()

        # Get movies already rated by the user if exclude_rated is True
        rated_movie_ids = set()
        if exclude_rated and ratings_df is not None:
            rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)

        # Predict ratings for all movies not rated by the user
        predictions = []
        for movie_id in all_movie_ids:
            if movie_id not in rated_movie_ids:
                predicted_rating = self.predict_rating(user_id, movie_id)
                if predicted_rating >= min_rating:
                    predictions.append((movie_id, predicted_rating))

        # Sort predictions by rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Get top N recommendations
        top_n_recs = predictions[:n]

        # Convert to DataFrame and add movie titles
        if not top_n_recs:
            return pd.DataFrame()

        recs_df = pd.DataFrame(top_n_recs, columns=['movieId', 'predicted_rating'])

        # Add movie titles if movies_df is available
        if self.movies_df is not None:
            recs_df = pd.merge(recs_df, self.movies_df[['movieId', 'title']],
                               on='movieId', how='left')

        return recs_df

    def save_model(self, model_path):
        """Save the model to disk"""
        if not self.trained:
            raise Exception("Model not trained. Call fit() first.")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'movie_factors': self.movie_factors,
                'user_ids': self.user_ids,
                'movie_ids': self.movie_ids,
                'mean_rating': self.mean_rating,
                'movies_df': self.movies_df,
                'n_factors': self.n_factors
            }, f)

    def load_model(self, model_path):
        """Load the model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.user_factors = model_data['user_factors']
        self.movie_factors = model_data['movie_factors']
        self.user_ids = model_data['user_ids']
        self.movie_ids = model_data['movie_ids']
        self.mean_rating = model_data['mean_rating']
        self.movies_df = model_data['movies_df']
        self.n_factors = model_data['n_factors']
        self.trained = True