import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import label_binarize
from collections import Counter
import os


class RecommenderPlots:
    """
    Utility class for generating visualizations for movie recommender system analysis.
    Creates plots for model evaluation, user behavior analysis, and recommendation visualizations.
    """

    def __init__(self, output_dir='./visualizations'):
        """
        Initialize the plotting utilities

        Parameters:
        -----------
        output_dir : str, optional (default='./visualizations')
            Directory to save plot images
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set default styling for plots
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_rating_distribution(self, ratings_df, save=True):
        """
        Plot the distribution of ratings

        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings with 'rating' column
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        plt.figure(figsize=(10, 6))

        # Count ratings by value
        rating_counts = ratings_df['rating'].value_counts().sort_index()

        # Create bar plot
        ax = sns.barplot(x=rating_counts.index, y=rating_counts.values)

        # Add count labels on bars
        for i, count in enumerate(rating_counts.values):
            ax.text(i, count + 0.1, f'{count:,}', ha='center')

        plt.title('Distribution of Movie Ratings', fontsize=16)
        plt.xlabel('Rating Value', fontsize=14)
        plt.ylabel('Number of Ratings', fontsize=14)
        plt.xticks(range(len(rating_counts.index)), rating_counts.index)

        if save:
            plt.savefig(f'{self.output_dir}/rating_distribution.png', bbox_inches='tight')

        plt.show()

    def plot_genre_distribution(self, movies_df, top_n=15, save=True):
        """
        Plot the distribution of movie genres

        Parameters:
        -----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with 'genres' column
        top_n : int, optional (default=15)
            Number of top genres to display
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Extract all genres
        all_genres = []
        for genres in movies_df['genres'].str.split('|'):
            all_genres.extend(genres)

        # Count genre occurrences
        genre_counts = Counter(all_genres)

        # Convert to DataFrame for plotting
        genre_df = pd.DataFrame({
            'genre': list(genre_counts.keys()),
            'count': list(genre_counts.values())
        })

        # Sort by count descending and get top N
        genre_df = genre_df.sort_values('count', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='count', y='genre', hue='genre', data=genre_df, palette='viridis', legend=False)

        # Add count labels
        for i, count in enumerate(genre_df['count']):
            ax.text(count + 10, i, f'{count:,}', va='center')

        plt.title('Distribution of Movie Genres (Top 15)', fontsize=16)
        plt.xlabel('Number of Movies', fontsize=14)
        plt.ylabel('Genre', fontsize=14)

        if save:
            plt.savefig(f'{self.output_dir}/genre_distribution.png', bbox_inches='tight')

        plt.show()

    def plot_recommendation_comparison(self, hybrid_recs, content_recs, collab_recs,
                                       top_n=10, save=True):
        """
        Compare recommendation scores from different recommender systems

        Parameters:
        -----------
        hybrid_recs : pandas.DataFrame
            DataFrame containing hybrid recommendations with 'title' and 'hybrid_score' columns
        content_recs : pandas.DataFrame
            DataFrame containing content-based recommendations with 'title' and 'content_score' columns
        collab_recs : pandas.DataFrame
            DataFrame containing collaborative filtering recommendations with 'title' and 'predicted_rating' columns
        top_n : int, optional (default=10)
            Number of top recommendations to display
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        plt.figure(figsize=(14, 10))

        # Prepare data for plotting
        hybrid_data = hybrid_recs.head(top_n)[['title', 'hybrid_score']].copy()
        hybrid_data = hybrid_data.sort_values('hybrid_score', ascending=True)

        # Plot horizontal bar chart
        ax = sns.barplot(x='hybrid_score', y='title', data=hybrid_data, color='purple', alpha=0.8)

        plt.title('Top Movie Recommendations (Hybrid Method)', fontsize=16)
        plt.xlabel('Recommendation Score', fontsize=14)
        plt.ylabel('Movie Title', fontsize=14)

        # Add score labels
        for i, score in enumerate(hybrid_data['hybrid_score']):
            ax.text(score + 0.01, i, f'{score:.2f}', va='center')

        if save:
            plt.savefig(f'{self.output_dir}/hybrid_recommendations.png', bbox_inches='tight')

        plt.show()

        # Create a comparison plot if all three recommendation types are available
        if not hybrid_recs.empty and not content_recs.empty and not collab_recs.empty:
            # Create a figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=False)

            # Plot hybrid recommendations
            hybrid_data = hybrid_recs.head(top_n)[['title', 'hybrid_score']].copy()
            hybrid_data = hybrid_data.sort_values('hybrid_score', ascending=True)
            sns.barplot(x='hybrid_score', y='title', data=hybrid_data, color='purple', alpha=0.8, ax=axes[0])
            axes[0].set_title('Hybrid Recommendations', fontsize=16)
            axes[0].set_xlabel('Hybrid Score', fontsize=14)
            axes[0].set_ylabel('Movie Title', fontsize=14)

            # Plot content-based recommendations
            content_data = content_recs.head(top_n)[['title', 'content_score']].copy()
            content_data = content_data.sort_values('content_score', ascending=True)
            sns.barplot(x='content_score', y='title', data=content_data, color='green', alpha=0.8, ax=axes[1])
            axes[1].set_title('Content-Based Recommendations', fontsize=16)
            axes[1].set_xlabel('Content Score', fontsize=14)
            axes[1].set_ylabel('Movie Title', fontsize=14)

            # Plot collaborative filtering recommendations
            collab_data = collab_recs.head(top_n)[['title', 'predicted_rating']].copy()
            collab_data = collab_data.sort_values('predicted_rating', ascending=True)
            sns.barplot(x='predicted_rating', y='title', data=collab_data, color='blue', alpha=0.8, ax=axes[2])
            axes[2].set_title('Collaborative Filtering Recommendations', fontsize=16)
            axes[2].set_xlabel('Predicted Rating', fontsize=14)
            axes[2].set_ylabel('Movie Title', fontsize=14)

            plt.tight_layout()

            if save:
                plt.savefig(f'{self.output_dir}/recommendation_comparison.png', bbox_inches='tight')

            plt.show()

    def plot_user_rating_behavior(self, ratings_df, n_users=10, save=True):
        """
        Plot rating behavior for a sample of users

        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings with 'userId' and 'rating' columns
        n_users : int, optional (default=10)
            Number of users to include in the visualization
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Get top N users by number of ratings
        top_users = ratings_df['userId'].value_counts().head(n_users).index

        # Filter ratings for these users
        user_ratings = ratings_df[ratings_df['userId'].isin(top_users)]

        # Calculate rating statistics for each user
        user_stats = []
        for user_id in top_users:
            user_data = ratings_df[ratings_df['userId'] == user_id]
            user_stats.append({
                'userId': user_id,
                'mean_rating': user_data['rating'].mean(),
                'std_rating': user_data['rating'].std(),
                'median_rating': user_data['rating'].median(),
                'count': len(user_data)
            })

        user_stats_df = pd.DataFrame(user_stats)

        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Plot 1: Distribution of ratings by user
        sns.boxplot(x='userId', y='rating', data=user_ratings, ax=ax1)
        ax1.set_title('Distribution of Ratings by User', fontsize=16)
        ax1.set_xlabel('User ID', fontsize=14)
        ax1.set_ylabel('Rating', fontsize=14)

        # Plot 2: Mean ratings with error bars
        sns.barplot(x='userId', y='mean_rating', data=user_stats_df, ax=ax2)
        ax2.errorbar(
            x=range(len(user_stats_df)),
            y=user_stats_df['mean_rating'],
            yerr=user_stats_df['std_rating'],
            fmt='none',
            color='black',
            capsize=5
        )
        ax2.set_title('Mean Ratings by User (with Standard Deviation)', fontsize=16)
        ax2.set_xlabel('User ID', fontsize=14)
        ax2.set_ylabel('Mean Rating', fontsize=14)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/user_rating_behavior.png', bbox_inches='tight')

        plt.show()

    def plot_model_evaluation(self, metrics_dict, save=True):
        """
        Plot evaluation metrics for recommender models

        Parameters:
        -----------
        metrics_dict : dict
            Dictionary containing model evaluation metrics
            Format: {'model_name': {'metric_name': value, ...}, ...}
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Extract metrics for plotting
        models = list(metrics_dict.keys())
        rmse_values = [metrics_dict[model].get('rmse', 0) for model in models]
        mae_values = [metrics_dict[model].get('mae', 0) for model in models]

        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot RMSE
        sns.barplot(x=models, y=rmse_values, ax=ax1)
        ax1.set_title('RMSE by Model', fontsize=16)
        ax1.set_xlabel('Model', fontsize=14)
        ax1.set_ylabel('RMSE (lower is better)', fontsize=14)

        # Add value labels
        for i, val in enumerate(rmse_values):
            ax1.text(i, val + 0.02, f'{val:.4f}', ha='center')

        # Plot MAE
        sns.barplot(x=models, y=mae_values, ax=ax2)
        ax2.set_title('MAE by Model', fontsize=16)
        ax2.set_xlabel('Model', fontsize=14)
        ax2.set_ylabel('MAE (lower is better)', fontsize=14)

        # Add value labels
        for i, val in enumerate(mae_values):
            ax2.text(i, val + 0.02, f'{val:.4f}', ha='center')

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/model_evaluation.png', bbox_inches='tight')

        plt.show()

    def plot_precision_recall_at_k(self, precision_values, recall_values, k_values, save=True):
        """
        Plot precision@k and recall@k metrics

        Parameters:
        -----------
        precision_values : list
            List of precision values at different k
        recall_values : list
            List of recall values at different k
        k_values : list
            List of k values used for evaluation
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        plt.figure(figsize=(12, 6))

        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot precision@k
        ax1.plot(k_values, precision_values, marker='o', linestyle='-', linewidth=2)
        ax1.set_title('Precision@k', fontsize=16)
        ax1.set_xlabel('k (number of recommendations)', fontsize=14)
        ax1.set_ylabel('Precision', fontsize=14)
        ax1.grid(True)

        # Plot recall@k
        ax2.plot(k_values, recall_values, marker='o', linestyle='-', linewidth=2, color='green')
        ax2.set_title('Recall@k', fontsize=16)
        ax2.set_xlabel('k (number of recommendations)', fontsize=14)
        ax2.set_ylabel('Recall', fontsize=14)
        ax2.grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/precision_recall_at_k.png', bbox_inches='tight')

        plt.show()

    def plot_similarity_matrix(self, similarity_matrix, movie_titles, top_n=20, save=True):
        """
        Plot a similarity matrix heatmap for movies

        Parameters:
        -----------
        similarity_matrix : numpy.ndarray
            Matrix of similarity scores between movies
        movie_titles : list
            List of movie titles corresponding to the similarity matrix
        top_n : int, optional (default=20)
            Number of top movies to include in the visualization
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Select top N movies
        if len(movie_titles) > top_n:
            # Use the first top_n movies
            similarity_matrix = similarity_matrix[:top_n, :top_n]
            movie_titles = movie_titles[:top_n]

        # Create heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=movie_titles,
            yticklabels=movie_titles
        )
        plt.title(f'Movie Similarity Matrix (Top {len(movie_titles)} Movies)', fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/similarity_matrix.png', bbox_inches='tight')

        plt.show()

    def plot_rating_trends(self, ratings_df, time_column='timestamp', save=True):
        """
        Plot trends in ratings over time

        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings with rating and timestamp columns
        time_column : str, optional (default='timestamp')
            Name of the column containing time information
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Ensure timestamp is datetime
        if pd.api.types.is_numeric_dtype(ratings_df[time_column]):
            ratings_df = ratings_df.copy()
            ratings_df['date'] = pd.to_datetime(ratings_df[time_column], unit='s')
        elif pd.api.types.is_datetime64_dtype(ratings_df[time_column]):
            ratings_df = ratings_df.copy()
            ratings_df['date'] = ratings_df[time_column]
        else:
            try:
                ratings_df = ratings_df.copy()
                ratings_df['date'] = pd.to_datetime(ratings_df[time_column])
            except:
                raise ValueError(f"Could not convert {time_column} to datetime format")

        # Extract year and month
        ratings_df['year_month'] = ratings_df['date'].dt.to_period('M')

        # Group by year_month and calculate statistics
        monthly_stats = ratings_df.groupby('year_month').agg(
            mean_rating=('rating', 'mean'),
            count=('rating', 'count')
        ).reset_index()

        # Convert year_month to datetime for plotting
        monthly_stats['date'] = monthly_stats['year_month'].dt.to_timestamp()

        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        # Plot average ratings over time
        ax1.plot(monthly_stats['date'], monthly_stats['mean_rating'], marker='o', linestyle='-', linewidth=2)
        ax1.set_title('Average Rating Over Time', fontsize=16)
        ax1.set_ylabel('Average Rating', fontsize=14)
        ax1.grid(True)

        # Plot number of ratings over time
        ax2.plot(monthly_stats['date'], monthly_stats['count'], marker='o', linestyle='-', linewidth=2, color='green')
        ax2.set_title('Number of Ratings Over Time', fontsize=16)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('Number of Ratings', fontsize=14)
        ax2.grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/rating_trends.png', bbox_inches='tight')

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, threshold=3.5, save=True):
        """
        Plot confusion matrix for rating predictions

        Parameters:
        -----------
        y_true : array-like
            True ratings
        y_pred : array-like
            Predicted ratings
        threshold : float, optional (default=3.5)
            Threshold to classify as positive/liked (>= threshold) or negative/disliked (< threshold)
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Convert ratings to binary (liked/disliked)
        y_true_binary = (np.array(y_true) >= threshold).astype(int)
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Disliked", "Liked"],
            yticklabels=["Disliked", "Liked"]
        )
        plt.title(f'Confusion Matrix (Threshold = {threshold})', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        if save:
            plt.savefig(f'{self.output_dir}/confusion_matrix.png', bbox_inches='tight')

        plt.show()

    def plot_recommendation_diversity(self, recommendations_df, movies_df, n_users=10, save=True):
        """
        Plot diversity of recommendations across users

        Parameters:
        -----------
        recommendations_df : pandas.DataFrame
            DataFrame containing recommendations with 'userId' and 'movieId' columns
        movies_df : pandas.DataFrame
            DataFrame containing movie information with 'movieId' and 'genres' columns
        n_users : int, optional (default=10)
            Number of users to include in the visualization
        save : bool, optional (default=True)
            Whether to save the plot to a file
        """
        # Process user recommendations
        user_ids = recommendations_df['userId'].unique()[:n_users]

        # Get genre distribution for each user's recommendations
        genre_distribution = {}

        for user_id in user_ids:
            user_recs = recommendations_df[recommendations_df['userId'] == user_id]
            user_movies = movies_df[movies_df['movieId'].isin(user_recs['movieId'])]

            # Extract genres
            all_genres = []
            for genres in user_movies['genres'].str.split('|'):
                all_genres.extend(genres)

            # Count genre occurrences
            genre_counts = Counter(all_genres)
            genre_distribution[user_id] = genre_counts

        # Get all unique genres
        all_genres = set()
        for counts in genre_distribution.values():
            all_genres.update(counts.keys())

        # Create a DataFrame for plotting
        genre_data = []
        for user_id, counts in genre_distribution.items():
            for genre in all_genres:
                genre_data.append({
                    'userId': user_id,
                    'genre': genre,
                    'count': counts.get(genre, 0)
                })

        genre_df = pd.DataFrame(genre_data)

        # Plot genre distribution by user
        plt.figure(figsize=(16, 12))
        sns.barplot(x='genre', y='count', hue='userId', data=genre_df)
        plt.title('Genre Distribution in Recommendations by User', fontsize=16)
        plt.xlabel('Genre', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=90)
        plt.legend(title='User ID')
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/recommendation_diversity.png', bbox_inches='tight')

        plt.show()


# Example usage:
if __name__ == "__main__":
    # Load data
    ratings_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/ratings_processed.csv")
    movies_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/movies_processed.csv")

    # Initialize plotting utilities
    plots = RecommenderPlots(output_dir='./visualizations')

    # Plot rating distribution
    plots.plot_rating_distribution(ratings_df)

    # Plot genre distribution
    plots.plot_genre_distribution(movies_df)

    # Plot user rating behavior
    plots.plot_user_rating_behavior(ratings_df)

    # Example of model evaluation metrics
    metrics = {
        'Content-Based': {'rmse': 0.95, 'mae': 0.75},
        'Collaborative': {'rmse': 0.88, 'mae': 0.71},
        'Hybrid': {'rmse': 0.85, 'mae': 0.69}
    }

    plots.plot_model_evaluation(metrics)