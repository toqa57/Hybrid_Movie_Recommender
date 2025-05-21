import pandas as pd
import numpy as np
import os
import pickle
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender


from models.content_based import ContentBasedRecommender  # Your existing import


class HybridRecommender:
    """
    Hybrid movie recommendation system that combines content-based and collaborative filtering approaches.
    """

    def __init__(self, content_weight=0.4, collab_weight=0.6):
        """
        Initialize the hybrid recommender system

        Parameters:
        -----------
        content_weight : float, optional (default=0.4)
            Weight assigned to content-based recommendations
        collab_weight : float, optional (default=0.6)
            Weight assigned to collaborative filtering recommendations
        """
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeFilteringRecommender()
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.movies_df = None

    def fit(self, ratings_df, movies_df):
        """
        Build both recommendation models

        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings
        movies_df : pandas.DataFrame
            DataFrame containing movie information
        """
        self.movies_df = movies_df.copy()

        # Train content-based model
        print("Training content-based model...")
        self.content_recommender.fit(movies_df)

        # Train collaborative filtering model
        print("\nTraining collaborative filtering model...")
        self.collab_recommender.fit(ratings_df, movies_df)

        print("\nHybrid recommender system trained successfully")

    def set_weights(self, content_weight, collab_weight):
        """
        Set weights for combining recommendations

        Parameters:
        -----------
        content_weight : float
            Weight assigned to content-based recommendations
        collab_weight : float
            Weight assigned to collaborative filtering recommendations
        """
        # Normalize weights
        total = content_weight + collab_weight
        self.content_weight = content_weight / total
        self.collab_weight = collab_weight / total

        print(f"Weights set to: Content-based = {self.content_weight:.2f}, Collaborative = {self.collab_weight:.2f}")

    def recommend(self, user_id, ratings_df, top_n=10, min_collab_rating=3.0):
        """
        Generate hybrid recommendations for a user

        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        ratings_df : pandas.DataFrame
            DataFrame containing all user ratings
        top_n : int, optional (default=10)
            Number of recommendations to return
        min_collab_rating : float, optional (default=3.0)
            Minimum predicted rating threshold for collaborative filtering

        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame containing recommended movies with hybrid scores
        """
        # Get user's ratings
        user_ratings = ratings_df[ratings_df['userId'] == user_id]

        # Get content-based recommendations (more than needed to ensure enough overlap)
        content_recs = self.content_recommender.recommend_for_user(user_ratings, top_n=top_n * 3)

        # Get collaborative filtering recommendations (more than needed to ensure enough overlap)
        collab_recs = self.collab_recommender.get_top_n_recommendations(
            user_id, n=top_n * 3, min_rating=min_collab_rating, exclude_rated=True, ratings_df=ratings_df
        )

        # Handle cases when one or both recommenders return empty
        if content_recs.empty and collab_recs.empty:
            print("No recommendations available from either model.")
            return pd.DataFrame()

        if content_recs.empty:
            print("No content-based recommendations available. Using only collaborative filtering.")
            return collab_recs.head(top_n)

        if collab_recs.empty:
            print("No collaborative filtering recommendations available. Using only content-based.")
            return content_recs.head(top_n)

        # Normalize scores for each approach
        content_recs['normalized_content_score'] = (
            content_recs['content_score'] / content_recs['content_score'].max()
        )

        collab_recs['normalized_collab_score'] = (
            (collab_recs['predicted_rating'] - 1) / 4.0  # Convert from 1-5 scale to 0-1
        )

        # Combine recommendations
        movie_scores = {}

        # Add content-based scores
        for _, row in content_recs.iterrows():
            movie_id = row['movieId']
            movie_scores[movie_id] = {
                'title': row['title'],
                'content_score': row['normalized_content_score'],
                'collab_score': 0.0,
                'hybrid_score': self.content_weight * row['normalized_content_score']
            }

        # Add collaborative filtering scores
        for _, row in collab_recs.iterrows():
            movie_id = row['movieId']
            collab_score = row['normalized_collab_score']
            if movie_id in movie_scores:
                movie_scores[movie_id]['collab_score'] = collab_score
                movie_scores[movie_id]['hybrid_score'] += self.collab_weight * collab_score
            else:
                # Movie not in content recs, add with 0 content score
                movie_scores[movie_id] = {
                    'title': row['title'],
                    'content_score': 0.0,
                    'collab_score': collab_score,
                    'hybrid_score': self.collab_weight * collab_score
                }

        # Convert to DataFrame and sort
        hybrid_df = pd.DataFrame.from_dict(movie_scores, orient='index')
        hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False)

        # Return top_n recommendations
        return hybrid_df.head(top_n).reset_index().rename(columns={'index': 'movieId'})
