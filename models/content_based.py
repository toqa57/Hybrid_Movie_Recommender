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
        self.movies_df = movies_df.drop_duplicates(subset='movieId').reset_index(drop=True)

        # Create index mapping from movieId to DataFrame index
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['movieId'])

        # Convert genres to a format suitable for TF-IDF
        genres_data = self.movies_df['genres'].str.replace('|', ' ', regex=False)

        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(genres_data)

        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        print(f"Content-based model built with {self.tfidf_matrix.shape[0]} movies")

    def get_similar_movies(self, movie_id, top_n=10):
        """
        Find movies similar to a given movie based on genres
        """
        if movie_id not in self.indices:
            print(f"Movie ID {movie_id} not found in the dataset")
            return []

        idx = self.indices[movie_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i[0] for i in sim_scores]
        similar_movies = []

        for i, score in enumerate(sim_scores):
            idx = movie_indices[i]
            movie_id_sim = self.movies_df.iloc[idx]['movieId']
            title = self.movies_df.iloc[idx]['title']
            similar_movies.append((movie_id_sim, title, score[1]))

        return similar_movies

    def recommend_for_movie(self, movie_id, top_n=10):
        """
        Generate content-based recommendations for a specific movie
        """
        similar_movies = self.get_similar_movies(movie_id, top_n)
        if not similar_movies:
            return pd.DataFrame()

        recommendations = pd.DataFrame(similar_movies, columns=['movieId', 'title', 'similarity_score'])
        return recommendations

    def recommend_for_user(self, user_ratings, top_n=10):
        """
        Generate content-based recommendations for a user based on their past ratings
        """
        if user_ratings.empty:
            return pd.DataFrame()

        rated_movies = user_ratings['movieId'].unique()
        movie_scores = {}

        for movie_id in rated_movies:
            if movie_id not in self.indices:
                continue

            rating_row = user_ratings[user_ratings['movieId'] == movie_id]
            if rating_row.empty:
                continue

            rating = rating_row['rating'].values[0]
            rating_weight = (rating - 1) / 4.0  # Normalize to 0-1

            similar_movies = self.get_similar_movies(movie_id, top_n=50)

            for sim_movie_id, _, sim_score in similar_movies:
                if sim_movie_id in rated_movies:
                    continue

                weighted_score = sim_score * rating_weight
                movie_scores[sim_movie_id] = max(movie_scores.get(sim_movie_id, 0), weighted_score)

        if not movie_scores:
            return pd.DataFrame()

        recommendations = pd.DataFrame({
            'movieId': list(movie_scores.keys()),
            'content_score': list(movie_scores.values())
        })

        recommendations = recommendations.sort_values('content_score', ascending=False).head(top_n)
        recommendations = pd.merge(recommendations,
                                   self.movies_df[['movieId', 'title']],
                                   on='movieId',
                                   how='left')

        return recommendations

    def save_model(self, model_path):
        """
        Save the content-based model to disk
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

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
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.movies_df = model_data['movies_df']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        self.vectorizer = model_data['vectorizer']

        print(f"Content-based model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    movies_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/movies_processed.csv")

    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movies_df)
    content_recommender.save_model('./models/content_based_model.pkl')

    movie_id = 1
    similar_movies = content_recommender.get_similar_movies(movie_id, top_n=10)

    print(f"\nMovies similar to {movies_df[movies_df['movieId'] == movie_id]['title'].values[0]}:")
    for movie_id, title, score in similar_movies:
        print(f"{title}: Similarity score {score:.4f}")

    ratings_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/ratings_processed.csv")
    user_id = 1
    user_ratings = ratings_df[ratings_df['userId'] == user_id]

    recommendations = content_recommender.recommend_for_user(user_ratings, top_n=10)
    print(f"\nTop 10 recommendations for User {user_id} based on content:")
    for _, row in recommendations.iterrows():
        print(f"{row['title']}: Score {row['content_score']:.4f}")
