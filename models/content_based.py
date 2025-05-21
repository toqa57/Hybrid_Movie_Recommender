import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


class ContentBasedRecommender:
    """
    Content-based filtering recommender system for movies.
    Uses TF-IDF and cosine similarity to find similar movies based on genres.
    """

    def __init__(self):
        """Initialize the content-based recommender system"""
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.vectorizer = None

    def fit(self, movies_df):
        """
        Build the content-based recommender model

        Parameters:
        -----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with at least 'movieId', 'title', and 'genres' columns
        """
        self.movies_df = movies_df.copy()

        # Create a numeric index for movie lookup
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['movieId'])

        # Convert genres to a format suitable for TF-IDF
        # Replace '|' with spaces to treat each genre as a separate word
        genres_data = self.movies_df['genres'].str.replace('|', ' ')

        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(genres_data)

        # Compute cosine similarity between movies
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        print(f"Content-based model built with {self.tfidf_matrix.shape[0]} movies")

    def get_similar_movies(self, movie_id, top_n=10):
        """
        Find movies similar to a given movie based on content (genres)

        Parameters:
        -----------
        movie_id : int
            ID of the movie to find similarities for
        top_n : int, optional (default=10)
            Number of similar movies to return

        Returns:
        --------
        similar_movies : list of tuples
            List of (movieId, title, similarity_score) tuples for similar movies
        """
        # Check if movie_id exists in our data
        if movie_id not in self.indices:
            print(f"Movie ID {movie_id} not found in the dataset")
            return []

        # Get the index of the movie
        idx = self.indices[movie_id]

        # Get similarity scores for all movies (pairwise)
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort movies by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N most similar movies (excluding the input movie itself)
        sim_scores = sim_scores[1:top_n + 1]

        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the similar movies with their similarity scores
        similar_movies = []
        for i, score in enumerate(sim_scores):
            idx = movie_indices[i]
            movie_id = self.movies_df.iloc[idx]['movieId']
            title = self.movies_df.iloc[idx]['title']
            similar_movies.append((movie_id, title, score[1]))

        return similar_movies

    def recommend_for_movie(self, movie_id, top_n=10):
        """
        Generate content-based recommendations for a specific movie

        Parameters:
        -----------
        movie_id : int
            ID of the movie to find recommendations for
        top_n : int, optional (default=10)
            Number of recommendations to return

        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame containing recommended movies and their similarity scores
        """
        similar_movies = self.get_similar_movies(movie_id, top_n)

        if not similar_movies:
            return pd.DataFrame()

        # Convert to DataFrame
        recommendations = pd.DataFrame(similar_movies, columns=['movieId', 'title', 'similarity_score'])

        return recommendations

    def recommend_for_user(self, user_ratings, top_n=10):
        """
        Generate content-based recommendations for a user based on their ratings

        Parameters:
        -----------
        user_ratings : pandas.DataFrame
            DataFrame containing user ratings with 'movieId' and 'rating' columns
        top_n : int, optional (default=10)
            Number of recommendations to return

        Returns:
        --------
        recommendations : pandas.DataFrame
            DataFrame containing recommended movies and their scores
        """
        # If user has no ratings, return empty recommendations
        if user_ratings.empty:
            return pd.DataFrame()

        # Get movies rated by the user
        rated_movies = user_ratings['movieId'].unique()

        # Create a dictionary to store movie scores
        movie_scores = {}

        # For each rated movie, find similar movies and add their scores
        for movie_id in rated_movies:
            # Skip if movie not in our content-based model
            if movie_id not in self.indices:
                continue

            # Get user's rating for this movie (normalized to 0-1 scale)
            rating = user_ratings[user_ratings['movieId'] == movie_id]['rating'].values[0]
            rating_weight = (rating - 1) / 4.0  # Normalize rating from 1-5 to 0-1

            # Get similar movies
            similar_movies = self.get_similar_movies(movie_id, top_n=50)  # Get more than needed to ensure diversity

            # Add scores to the dictionary, weighted by user's rating
            for sim_movie_id, _, sim_score in similar_movies:
                # Skip movies already rated by the user
                if sim_movie_id in rated_movies:
                    continue

                # Calculate weighted score
                weighted_score = sim_score * rating_weight

                # Add to movie_scores (use max if movie already in dictionary)
                if sim_movie_id in movie_scores:
                    movie_scores[sim_movie_id] = max(movie_scores[sim_movie_id], weighted_score)
                else:
                    movie_scores[sim_movie_id] = weighted_score

        # Convert scores to DataFrame and sort
        if not movie_scores:
            return pd.DataFrame()

        recommendations = pd.DataFrame({
            'movieId': list(movie_scores.keys()),
            'content_score': list(movie_scores.values())
        })

        # Sort by score (descending)
        recommendations = recommendations.sort_values('content_score', ascending=False)

        # Get top N recommendations
        recommendations = recommendations.head(top_n)

        # Add movie titles
        recommendations = pd.merge(recommendations,
                                   self.movies_df[['movieId', 'title']],
                                   on='movieId',
                                   how='left')

        return recommendations

    def save_model(self, model_path):
        """
        Save the content-based model to disk

        Parameters:
        -----------
        model_path : str
            Path to save the model
        """
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'movies_df': self.movies_df,
                'tfidf_matrix': self.tfidf_matrix,
                'cosine_sim': self.cosine_sim,
                'indices': self.indices,
                'vectorizer': self.vectorizer
            }, f)

        print(f"Content-based model saved to {model_path}")

    def load_model(self, model_path):
        """
        Load the content-based model from disk

        Parameters:
        -----------
        model_path : str
            Path to load the model from
        """
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.movies_df = model_data['movies_df']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        self.vectorizer = model_data['vectorizer']

        print(f"Content-based model loaded from {model_path}")


# Example usage:
if __name__ == "__main__":
    # Load preprocessed movies data
    movies_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/movies_processed.csv")

    # Initialize and build content-based recommender
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movies_df)

    # Save the model
    content_recommender.save_model('./models/content_based_model.pkl')

    # Example: Get similar movies for a specific movie
    movie_id = 1  # Example movie ID
    similar_movies = content_recommender.get_similar_movies(movie_id, top_n=10)

    print(f"\nMovies similar to {movies_df[movies_df['movieId'] == movie_id]['title'].values[0]}:")
    for movie_id, title, score in similar_movies:
        print(f"{title}: Similarity score {score:.4f}")

    # Example: Load user ratings and recommend movies
    ratings_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/ratings_processed.csv")
    user_id = 1  # Example user ID
    user_ratings = ratings_df[ratings_df['userId'] == user_id]

    recommendations = content_recommender.recommend_for_user(user_ratings, top_n=10)

    print(f"\nTop 10 recommendations for User {user_id} based on content:")
    for _, row in recommendations.iterrows():
        print(f"{row['title']}: Score {row['content_score']:.4f}")