import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_datasets(movies_path= "E:/intelligent programming/hybrid_movie_recommender/data/movies.csv"
                  , ratings_path= "E:/intelligent programming/hybrid_movie_recommender/data/ratings.csv"):
    """
    Load the MovieLens datasets (movies and ratings only) from specific CSV files

    Parameters:
    -----------
    movies_path : str
        Path to the movies CSV file
    ratings_path : str
        Path to the ratings CSV file

    Returns:
    --------
    movies_df : pandas.DataFrame
        DataFrame containing movie information
    ratings_df : pandas.DataFrame
        DataFrame containing user ratings
    """
    print("Loading datasets...")

    # Load datasets from CSV files
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    print(f"Movies dataset shape: {movies_df.shape}")
    print(f"Ratings dataset shape: {ratings_df.shape}")

    return movies_df, ratings_df


def clean_movies_data(movies_df):
    """
    Clean and preprocess the movies dataset

    Parameters:
    -----------
    movies_df : pandas.DataFrame
        Raw movies DataFrame

    Returns:
    --------
    movies_df : pandas.DataFrame
        Cleaned movies DataFrame
    """
    print("Cleaning movies dataset...")

    # Create a copy to avoid modifying the original dataframe
    movies_df = movies_df.copy()

    # Ensure column names are consistent
    if 'movie_id' in movies_df.columns:
        movies_df.rename(columns={'movie_id': 'movieId'}, inplace=True)
    if 'movie id' in movies_df.columns:
        movies_df.rename(columns={'movie id': 'movieId'}, inplace=True)

    # Check for duplicate movieIds
    duplicate_movies = movies_df[movies_df.duplicated(subset=['movieId'], keep=False)]
    if not duplicate_movies.empty:
        print(f"Warning: Found {len(duplicate_movies)} duplicate movieIds")
        # Keep the first occurrence of each movieId
        movies_df.drop_duplicates(subset=['movieId'], keep='first', inplace=True)

    # Ensure movieId is integer
    movies_df['movieId'] = movies_df['movieId'].astype(int)

    # Extract year from title if it exists (e.g., "Toy Story (1995)" -> 1995)
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')

    # Clean title (remove year)
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()

    # Handle missing genres
    if movies_df['genres'].isna().sum() > 0:
        print(f"Warning: Found {movies_df['genres'].isna().sum()} movies with missing genres")
        movies_df['genres'] = movies_df['genres'].fillna('Unknown')

    # Ensure genres are consistent (sometimes '|' or ',' or other separators are used)
    # Convert to a standard format using '|' as separator
    movies_df['genres'] = movies_df['genres'].str.replace(',', '|', regex=False)
    movies_df['genres'] = movies_df['genres'].str.replace(' - ', '|', regex=False)

    # Convert genres to lowercase for consistency
    movies_df['genres'] = movies_df['genres'].str.lower()

    return movies_df


def clean_ratings_data(ratings_df):
    """
    Clean and preprocess the ratings dataset

    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        Raw ratings DataFrame

    Returns:
    --------
    ratings_df : pandas.DataFrame
        Cleaned ratings DataFrame
    """
    print("Cleaning ratings dataset...")

    # Create a copy to avoid modifying the original dataframe
    ratings_df = ratings_df.copy()

    # Ensure column names are consistent
    if 'user_id' in ratings_df.columns:
        ratings_df.rename(columns={'user_id': 'userId'}, inplace=True)
    if 'movie_id' in ratings_df.columns:
        ratings_df.rename(columns={'movie_id': 'movieId'}, inplace=True)

    # Ensure userId and movieId are integers
    ratings_df['userId'] = ratings_df['userId'].astype(int)
    ratings_df['movieId'] = ratings_df['movieId'].astype(int)

    # Handle missing ratings
    if ratings_df['rating'].isna().sum() > 0:
        print(f"Warning: Found {ratings_df['rating'].isna().sum()} ratings with missing values")
        # For missing ratings, we can either drop them or impute with mean rating
        # Here we choose to drop them
        ratings_df = ratings_df.dropna(subset=['rating'])

    # Ensure rating is float
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    # Convert timestamp to datetime if it exists
    if 'timestamp' in ratings_df.columns:
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

    return ratings_df


def merge_datasets(movies_df, ratings_df):
    """
    Merge movies and ratings datasets

    Parameters:
    -----------
    movies_df : pandas.DataFrame
        Cleaned movies DataFrame
    ratings_df : pandas.DataFrame
        Cleaned ratings DataFrame

    Returns:
    --------
    merged_df : pandas.DataFrame
        Merged DataFrame containing movie information and ratings
    """
    print("Merging datasets...")

    # Check for movies in ratings that don't exist in movies_df
    ratings_movies = set(ratings_df['movieId'].unique())
    movies_list = set(movies_df['movieId'].unique())

    missing_movies = ratings_movies - movies_list
    if missing_movies:
        print(f"Warning: {len(missing_movies)} movies in ratings do not exist in movies dataset")
        print("Filtering out ratings for non-existent movies")
        ratings_df = ratings_df[ratings_df['movieId'].isin(movies_list)]

    # Merge datasets on movieId
    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')

    print(f"Merged dataset shape: {merged_df.shape}")

    return merged_df


def split_data(merged_df, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets

    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Merged DataFrame containing movie information and ratings
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Random seed for reproducibility

    Returns:
    --------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Testing dataset
    """
    print("Splitting data into training and testing sets...")

    # Split based on users to make sure we have ratings from all users in both sets
    train_df, test_df = train_test_split(
        merged_df,
        test_size=test_size,
        random_state=random_state,
        stratify=merged_df['userId'].apply(lambda x: min(x, 100))
        # Stratify by userId, capped at 100 to avoid too many categories
    )

    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

    return train_df, test_df


def save_processed_data(movies_df, ratings_df, merged_df, train_df, test_df, output_path):
    """
    Save processed datasets to CSV files

    Parameters:
    -----------
    movies_df : pandas.DataFrame
        Cleaned movies DataFrame
    ratings_df : pandas.DataFrame
        Cleaned ratings DataFrame
    merged_df : pandas.DataFrame
        Merged DataFrame
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Testing dataset
    output_path : str
        Path to save the processed datasets
    """
    print("Saving processed datasets...")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save processed datasets
    movies_df.to_csv(os.path.join(output_path, 'movies_processed.csv'), index=False)
    ratings_df.to_csv(os.path.join(output_path, 'ratings_processed.csv'), index=False)
    merged_df.to_csv(os.path.join(output_path, 'merged_data.csv'), index=False)
    train_df.to_csv(os.path.join(output_path, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_data.csv'), index=False)

    print(f"Processed datasets saved to {output_path}")


def analyze_data(movies_df, ratings_df, merged_df):
    """
    Generate basic statistics and insights about the data

    Parameters:
    -----------
    movies_df : pandas.DataFrame
        Cleaned movies DataFrame
    ratings_df : pandas.DataFrame
        Cleaned ratings DataFrame
    merged_df : pandas.DataFrame
        Merged DataFrame
    """
    print("\nData Analysis:")

    # Movies dataset analysis
    print("\nMovies Dataset:")
    print(f"Total number of movies: {len(movies_df)}")

    # Extract unique genres
    all_genres = []
    for genres in movies_df['genres'].str.split('|'):
        all_genres.extend(genres)
    unique_genres = sorted(set(all_genres))

    print(f"Number of unique genres: {len(unique_genres)}")
    print(f"Most common genres: {pd.Series(all_genres).value_counts().head(5).to_dict()}")

    if 'year' in movies_df.columns:
        year_counts = movies_df['year'].value_counts().sort_index()
        print(f"Year range: {year_counts.index.min()} - {year_counts.index.max()}")
        print(f"Top 5 years with most movies: {year_counts.sort_values(ascending=False).head(5).to_dict()}")

    # Ratings dataset analysis
    print("\nRatings Dataset:")
    print(f"Total number of ratings: {len(ratings_df)}")
    print(f"Number of unique users: {ratings_df['userId'].nunique()}")
    print(f"Number of unique movies rated: {ratings_df['movieId'].nunique()}")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    print(f"Rating distribution: {ratings_df['rating'].value_counts().sort_index().to_dict()}")

    # Calculate rating statistics per user
    user_rating_counts = ratings_df.groupby('userId')['rating'].count()
    print(f"Average ratings per user: {user_rating_counts.mean():.2f}")
    print(f"Min ratings per user: {user_rating_counts.min()}")
    print(f"Max ratings per user: {user_rating_counts.max()}")

    # Calculate rating statistics per movie
    movie_rating_counts = ratings_df.groupby('movieId')['rating'].count()
    print(f"Average ratings per movie: {movie_rating_counts.mean():.2f}")
    print(f"Min ratings per movie: {movie_rating_counts.min()}")
    print(f"Max ratings per movie: {movie_rating_counts.max()}")

    # Merged dataset analysis
    print("\nMerged Dataset:")
    print(f"Shape: {merged_df.shape}")

    # Find top-rated movies (with at least 50 ratings)
    popular_movies = merged_df.groupby(['movieId', 'title']).agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    popular_movies = popular_movies[popular_movies['num_ratings'] >= 50].sort_values(
        by='avg_rating', ascending=False
    )

    print("\nTop 10 rated movies (with at least 50 ratings):")
    for _, row in popular_movies.head(10).iterrows():
        print(f"{row['title']}: {row['avg_rating']:.2f} (based on {row['num_ratings']} ratings)")


def main():
    """
    Main function to execute the data preprocessing pipeline
    """
    # Set paths to your CSV files - update these with your actual file paths
    movies_path = "E:/intelligent programming/hybrid_movie_recommender/data/movies.csv"  # Update with your movies.csv file path
    ratings_path = "E:/intelligent programming/hybrid_movie_recommender/data/ratings.csv"  # Update with your ratings.csv file path
    output_path = './processed_data'

    # Load datasets
    movies_df, ratings_df = load_datasets(movies_path, ratings_path)

    # Clean datasets
    movies_df = clean_movies_data(movies_df)
    ratings_df = clean_ratings_data(ratings_df)

    # Merge datasets
    merged_df = merge_datasets(movies_df, ratings_df)

    # Split data
    train_df, test_df = split_data(merged_df)

    # Analyze data
    analyze_data(movies_df, ratings_df, merged_df)

    # Save processed data
    save_processed_data(movies_df, ratings_df, merged_df, train_df, test_df, output_path)

    print("\nData preprocessing completed successfully!")


if __name__ == "__main__":
    main()