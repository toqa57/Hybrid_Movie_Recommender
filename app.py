# app.py
import streamlit as st
import pandas as pd
import numpy as np
from models.hybrid_model import HybridRecommender
import time
import os

# Set page config
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load data function with caching
@st.cache_data
def load_data():
    # Update these paths to match your actual data locations
    train_df = pd.read_csv("data/processed_data/train_data.csv")
    test_df = pd.read_csv("data/processed_data/test_data.csv")
    movies_df = pd.read_csv("data/processed_data/movies_processed.csv")
    return train_df, test_df, movies_df


# Initialize and train the recommender with caching
@st.cache_resource
def initialize_recommender(train_df, movies_df, content_weight=0.4, collab_weight=0.6):
    recommender = HybridRecommender(content_weight=content_weight, collab_weight=collab_weight)
    recommender.fit(train_df, movies_df)
    return recommender


def main():
    st.title("🎬 Hybrid Movie Recommender System")
    st.markdown("""
    This app provides personalized movie recommendations using a hybrid approach 
    combining content-based and collaborative filtering techniques.
    """)

    # Load data
    try:
        train_df, test_df, movies_df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")

        # Weight sliders
        content_weight = st.slider(
            "Content-based weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1
        )
        collab_weight = st.slider(
            "Collaborative filtering weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )

        # User selection
        user_id = st.selectbox(
            "Select a user ID",
            options=sorted(train_df['userId'].unique()),
            index=0
        )

        # Number of recommendations
        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10
        )

        # Minimum collaborative rating threshold
        min_collab_rating = st.slider(
            "Minimum collaborative rating threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )

    # Initialize recommender
    recommender = initialize_recommender(
        train_df,
        movies_df,
        content_weight=content_weight,
        collab_weight=collab_weight
    )

    # Main content area
    st.header(f"Top {top_n} Recommendations for User {user_id}")

    # Get recommendations
    with st.spinner("Generating recommendations..."):
        try:
            recommendations = recommender.recommend(
                user_id=user_id,
                ratings_df=train_df,
                top_n=top_n,
                min_collab_rating=min_collab_rating
            )

            if not recommendations.empty:
                # Display recommendations in columns
                cols = st.columns(3)
                for i, row in recommendations.iterrows():
                    with cols[i % 3]:
                        with st.container(border=True):
                            st.subheader(f"{i + 1}. {row['title']}")

                            # Check which scores are available in the output
                            if 'hybrid_score' in recommendations.columns:
                                st.write(f"**Hybrid Score:** {row['hybrid_score']:.2f}")
                            if 'content_score' in recommendations.columns:
                                st.write(f"**Content Score:** {row['content_score']:.2f}")
                            if 'collab_score' in recommendations.columns:
                                st.write(f"**Collaborative Score:** {row['collab_score']:.2f}")
                            if 'predicted_rating' in recommendations.columns:
                                st.write(f"**Predicted Rating:** {row['predicted_rating']:.2f}")

                            # Get movie details from movies_df
                            movie_details = movies_df[movies_df['movieId'] == row['movieId']].iloc[0]

                            if 'genres' in movie_details:
                                st.write(f"**Genres:** {movie_details['genres']}")
                            if 'year' in movie_details:
                                st.write(f"**Year:** {movie_details['year']}")
            else:
                st.warning("No recommendations could be generated for this user.")

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")


if __name__ == "__main__":
    main()
