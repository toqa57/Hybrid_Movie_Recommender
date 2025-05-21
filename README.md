# Hybrid_Movie_Recommender
# README.md 

* Hybrid Movie Recommendation System
A comprehensive movie recommendation system that combines content-based filtering and collaborative filtering approaches to provide personalized movie recommendations.

* Project Overview
This project implements a hybrid recommendation system for movies using the MovieLens 100K dataset. The system uses two main recommendation approaches:

1- Content-Based Filtering: Recommends movies based on their genre similarity
2- Collaborative Filtering: Recommends movies based on user rating patterns
3- Hybrid Approach: Combines both methods with configurable weights

# Features

.Data preprocessing and exploration
. Content-based filtering using TF-IDF and cosine similarity
. Collaborative filtering using Singular Value Decomposition (SVD)
. Hybrid recommendation engine
. Interactive Streamlit web interface
. Model evaluation and metrics
. User and movie selection
. Visualizations and charts

# Directory Structure
movie_recommendation_system/
├── app.py                   # Main Streamlit application
├── main.py                  # Entry point for the application
├── requirements.txt         # Project dependencies
├── data/                    # Data handling
│   ├── __init__.py
│   └── loader.py            # Data loading and preprocessing
├── models/                  # Recommendation models
│   ├── __init__.py
│   ├── content_based.py     # Content-based filtering
│   ├── collaborative.py     # Collaborative filtering
│   └── hybrid.py            # Hybrid recommendation
├── visualization/           # Visualization utilities
│   ├── __init__.py
│   └── plots.py             # Plotting functions
└── utils/                   # Utility functions
    ├── __init__.py
    └── evaluation.py        # Evaluation metrics
    
# Installation

1- Clone the repository:

git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

2- Install dependencies:

pip install -r requirements.txt

3- Run the application:

python main.py
Alternatively, you can run the Streamlit app directly:
streamlit run app.py

# Dataset
The project uses the MovieLens 100K dataset, which includes:

* 100,000 ratings (1-5) from 943 users on 1,682 movies
* User demographic information (age, gender, occupation)
* Movie information (title, genres)

The dataset will be automatically downloaded when you run the application.
# Usage

* Run the application
* Select a user from the dropdown menu
* Select a movie to base recommendations on
* Adjust the collaborative filtering weight slider (0 = pure content-based, 1 = pure collaborative)
* Select the number of recommendations to generate
* Click "Get Recommendations" to see personalized movie suggestions

# Evaluation Metrics
The system evaluates the recommendations using:

* Root Mean Square Error (RMSE)
* Mean Absolute Error (MAE)
* Comparison with baseline metrics

# Requirements

Python 3.8+
Streamlit
Pandas
NumPy
Scikit-learn
Surprise
Matplotlib
Seaborn
Requests

# License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

# MovieLens for providing the dataset
Streamlit for the web application framework
Surprise for the collaborative filtering implementation
