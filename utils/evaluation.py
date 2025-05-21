import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


def calculate_recommendation_metrics(recommendations, ground_truth, k=10):
    """
    Calculate recommendation-specific evaluation metrics

    Parameters:
    -----------
    recommendations : list
        List of recommended movieIds
    ground_truth : list
        List of relevant movieIds (usually positively rated by user)
    k : int, optional (default=10)
        Number of recommendations to consider

    Returns:
    --------
    metrics : dict
        Dictionary containing metrics (precision@k, recall@k, F1@k)
    """
    # Ensure we only consider top k recommendations
    recommendations = recommendations[:k]

    # Find the number of relevant recommendations
    relevant_and_recommended = set(recommendations).intersection(set(ground_truth))
    num_relevant_and_recommended = len(relevant_and_recommended)

    # Calculate precision@k
    precision = num_relevant_and_recommended / len(recommendations) if recommendations else 0

    # Calculate recall@k
    recall = num_relevant_and_recommended / len(ground_truth) if ground_truth else 0

    # Calculate F1@k
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1
    }


def calculate_ranking_metrics(recommendations, ground_truth, ratings=None):
    """
    Calculate ranking-based evaluation metrics

    Parameters:
    -----------
    recommendations : list of tuples
        List of (movieId, score) tuples, ordered by recommendation score
    ground_truth : dict
        Dictionary mapping movieId to rating
    ratings : dict, optional
        Dictionary mapping movieId to user ratings (for NDCG)

    Returns:
    --------
    metrics : dict
        Dictionary containing metrics (MAP, MRR, NDCG)
    """
    if not recommendations or not ground_truth:
        return {'MAP': 0, 'MRR': 0, 'NDCG': 0}

    # Mean Average Precision (MAP)
    ap = 0
    hits = 0

    # Mean Reciprocal Rank (MRR)
    mrr = 0

    # Normalized Discounted Cumulative Gain (NDCG)
    dcg = 0
    idcg = 0

    # Calculate ideal DCG (IDCG)
    if ratings:
        # Sort ground truth by rating for IDCG
        sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        for i, (_, rating) in enumerate(sorted_ratings):
            # IDCG calculation assumes ratings are sorted in descending order
            idcg += (2 ** rating - 1) / np.log2(i + 2)  # +2 because i is 0-indexed and log2(1) = 0

    for i, (movie_id, _) in enumerate(recommendations):
        # Check if movie is relevant
        if movie_id in ground_truth:
            hits += 1
            ap += hits / (i + 1)

            # MRR is the reciprocal of the rank of the first relevant item
            if mrr == 0:
                mrr = 1 / (i + 1)

            # DCG calculation
            if ratings and movie_id in ratings:
                rating = ratings[movie_id]
                dcg += (2 ** rating - 1) / np.log2(i + 2)

    # Finalize MAP
    map_score = ap / len(ground_truth) if ground_truth else 0

    # Finalize NDCG
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        'MAP': map_score,
        'MRR': mrr,
        'NDCG': ndcg
    }


def evaluate_recommender(recommender, test_data, train_data=None, k=10):
    """
    Comprehensive evaluation of a recommender system

    Parameters:
    -----------
    recommender : object
        Recommender system object with a recommend method
    test_data : pandas.DataFrame
        Test data containing userId, movieId, rating columns
    train_data : pandas.DataFrame, optional
        Training data used to generate recommendations
    k : int, optional (default=10)
        Number of recommendations to consider

    Returns:
    --------
    eval_results : dict
        Dictionary containing evaluation results
    """
    # Get unique users in test data
    users = test_data['userId'].unique()

    # Metrics to track
    all_precision = []
    all_recall = []
    all_f1 = []
    all_map = []
    all_mrr = []
    all_ndcg = []

    # For each user
    for user_id in users:
        # Get user's test ratings
        user_test = test_data[test_data['userId'] == user_id]

        # Define ground truth as items with ratings >= 4
        ground_truth = user_test[user_test['rating'] >= 4]['movieId'].tolist()

        # If no positive ratings in test set, skip this user
        if not ground_truth:
            continue

        # Get recommendations for this user
        if train_data is not None:
            # Use training data to generate recommendations
            user_train = train_data[train_data['userId'] == user_id]
            recommendations = recommender.recommend(user_id, user_train, top_n=k)
        else:
            # If no training data provided, assume recommender can handle this
            recommendations = recommender.recommend(user_id, top_n=k)

        # Extract movie IDs from recommendations
        if isinstance(recommendations, pd.DataFrame):
            rec_movies = recommendations['movieId'].tolist()
            rec_scores = recommendations[['movieId', 'hybrid_score']].values.tolist()
        else:
            # If recommendations are already a list
            rec_movies = [r[0] for r in recommendations[:k]]
            rec_scores = recommendations[:k]

        # Calculate basic metrics
        basic_metrics = calculate_recommendation_metrics(rec_movies, ground_truth, k)
        all_precision.append(basic_metrics['precision@k'])
        all_recall.append(basic_metrics['recall@k'])
        all_f1.append(basic_metrics['f1@k'])

        # Create ground truth and ratings dictionaries for ranking metrics
        gt_dict = {row['movieId']: 1 for _, row in user_test[user_test['rating'] >= 4].iterrows()}
        ratings_dict = {row['movieId']: row['rating'] for _, row in user_test.iterrows()}

        # Calculate ranking metrics
        ranking_metrics = calculate_ranking_metrics(rec_scores, gt_dict, ratings_dict)
        all_map.append(ranking_metrics['MAP'])
        all_mrr.append(ranking_metrics['MRR'])
        all_ndcg.append(ranking_metrics['NDCG'])

    # Aggregate results
    eval_results = {
        'precision@k': np.mean(all_precision) if all_precision else 0,
        'recall@k': np.mean(all_recall) if all_recall else 0,
        'f1@k': np.mean(all_f1) if all_f1 else 0,
        'MAP': np.mean(all_map) if all_map else 0,
        'MRR': np.mean(all_mrr) if all_mrr else 0,
        'NDCG': np.mean(all_ndcg) if all_ndcg else 0
    }

    return eval_results


def evaluate_rating_prediction(predictions, actual):
    """
    Evaluate rating prediction performance

    Parameters:
    -----------
    predictions : list or array-like
        Predicted ratings
    actual : list or array-like
        Actual ratings

    Returns:
    --------
    metrics : dict
        Dictionary containing metrics (RMSE, MAE)
    """
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(actual, predictions))

    # Calculate MAE
    mae = mean_absolute_error(actual, predictions)

    return {
        'RMSE': rmse,
        'MAE': mae
    }


def compare_recommenders(recommenders, test_data, train_data=None, k=10):
    """
    Compare multiple recommender systems

    Parameters:
    -----------
    recommenders : dict
        Dictionary mapping recommender names to recommender objects
    test_data : pandas.DataFrame
        Test data for evaluation
    train_data : pandas.DataFrame, optional
        Training data for generating recommendations
    k : int, optional (default=10)
        Number of recommendations to consider

    Returns:
    --------
    comparison : pandas.DataFrame
        DataFrame containing evaluation metrics for each recommender
    """
    results = {}

    for name, recommender in recommenders.items():
        print(f"Evaluating {name}...")
        eval_results = evaluate_recommender(recommender, test_data, train_data, k)
        results[name] = eval_results

    # Convert to DataFrame for easy comparison
    comparison = pd.DataFrame(results).T

    return comparison


def hybrid_evaluate(self, test_df, train_df, k=10):
    """
    Evaluate the hybrid recommender system

    Parameters:
    -----------
    test_df : pandas.DataFrame
        DataFrame containing test ratings
    train_df : pandas.DataFrame
        DataFrame containing training ratings
    k : int, optional (default=10)
        Number of recommendations to consider for evaluation metrics

    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # We'll need to compute RMSE and MAE ourselves since the collab_recommender doesn't have evaluate()
    # First, get predictions for all user-item pairs in the test set
    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']

        # Get the predicted rating
        try:
            # You might need to adjust this based on your actual implementation
            predicted_rating = self.predict_rating(user_id, movie_id, train_df)

            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        except:
            # Skip if prediction fails for this user-movie pair
            continue

    # Calculate RMSE and MAE
    rmse = sqrt(mean_squared_error(actuals, predictions)) if predictions else 0
    mae = mean_absolute_error(actuals, predictions) if predictions else 0

    # Get unique users in test set
    test_users = test_df['userId'].unique()

    # Calculate precision@k, recall@k, and F1@k for each user
    precision_at_k = []
    recall_at_k = []
    f1_at_k = []

    for user_id in test_users:
        # Get user's test ratings
        user_test_ratings = test_df[test_df['userId'] == user_id]

        # Get movies rated 4 or above by the user in the test set (relevant items)
        relevant_movies = set(user_test_ratings[user_test_ratings['rating'] >= 4]['movieId'])

        # If no relevant movies in test set, skip this user
        if not relevant_movies:
            continue

        # Generate recommendations for this user
        recommendations = self.recommend(user_id, train_df, top_n=k)

        # If no recommendations were generated, skip this user
        if recommendations.empty:
            continue

        # Get recommended movie IDs
        recommended_movies = set(recommendations['movieId'])

        # Calculate precision@k: What percentage of our recommendations were relevant?
        precision = len(relevant_movies.intersection(recommended_movies)) / len(recommended_movies)
        precision_at_k.append(precision)

        # Calculate recall@k: What percentage of relevant items did we recommend?
        recall = len(relevant_movies.intersection(recommended_movies)) / len(relevant_movies)
        recall_at_k.append(recall)

        # Calculate F1@k: Harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_at_k.append(f1)

    # Compute average metrics
    avg_precision = np.mean(precision_at_k) if precision_at_k else 0
    avg_recall = np.mean(recall_at_k) if recall_at_k else 0
    avg_f1 = np.mean(f1_at_k) if f1_at_k else 0

    # Combine metrics
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'Precision@k': avg_precision,
        'Recall@k': avg_recall,
        'F1@k': avg_f1
    }

    print(f"Evaluation results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


# Definition for HybridRecommender.predict_rating method
def predict_rating(self, user_id, movie_id, train_df=None):
    """
    Predict a rating for a specific user-movie pair

    Parameters:
    -----------
    user_id : int
        ID of the user
    movie_id : int
        ID of the movie
    train_df : pandas.DataFrame, optional
        Training data to use for generating predictions

    Returns:
    --------
    rating : float
        Predicted rating for the user-movie pair
    """
    # Get collaborative filtering prediction
    collab_pred = self.collab_recommender.predict_rating(user_id, movie_id)

    # Get content-based prediction if available
    try:
        content_pred = self.content_recommender.predict_rating(user_id, movie_id, train_df)
    except:
        # If content prediction fails, just use collaborative prediction
        return collab_pred

    # Hybrid prediction is a weighted combination
    hybrid_pred = (self.collab_weight * collab_pred) + (self.content_weight * content_pred)

    # Ensure rating is within valid range (1-5)
    hybrid_pred = max(1.0, min(5.0, hybrid_pred))

    return hybrid_pred


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from models.hybrid_model import HybridRecommender

    # Load data
    train_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/train_data.csv")
    test_df = pd.read_csv("E:/intelligent programming/hybrid_movie_recommender/data/processed_data/test_data.csv")
    movies_df = pd.read_csv(
        "E:/intelligent programming/hybrid_movie_recommender/data/processed_data/movies_processed.csv")

    # Create and train hybrid recommender
    hybrid_rec = HybridRecommender(content_weight=0.4, collab_weight=0.6)
    hybrid_rec.fit(train_df, movies_df)

    # Evaluate the recommender
    metrics = hybrid_evaluate(hybrid_rec, test_df, train_df, k=10)

    print("\nEvaluation results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
