import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tmdbv3api import TMDb, Movie

# TMDB API setup
tmdb = TMDb()
tmdb.api_key = '76f7e24790461fdca4f863b0930273eb'
movie = Movie()

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
@st.cache_data
def load_data(file_path):
    movies_df = pd.read_csv(file_path)
    return movies_df

file_path = "D:/CHROME DOWNLOAD/movies.csv"
movies_df = load_data(file_path)

# Extract relevant columns and preprocess
def preprocess_data(movies_df):
    relevant_columns = ['title', 'genres', 'overview', 'vote_average', 'vote_count', 'cast', 'crew', 'keywords']
    movies_cleaned_df = movies_df[relevant_columns].copy()

    # Handle missing values
    movies_cleaned_df['overview'].fillna('', inplace=True)
    movies_cleaned_df['genres'].fillna('', inplace=True)
    movies_cleaned_df['cast'].fillna('', inplace=True)
    movies_cleaned_df['crew'].fillna('', inplace=True)
    movies_cleaned_df['keywords'].fillna('', inplace=True)
    movies_cleaned_df['vote_average'].fillna(0, inplace=True)
    movies_cleaned_df['vote_count'].fillna(0, inplace=True)

    # Perform sentiment analysis
    sid = SentimentIntensityAnalyzer()
    movies_cleaned_df['sentiment'] = movies_cleaned_df['overview'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Convert stringified lists into actual lists
    def convert_to_list(column):
        return column.apply(lambda x: x.split() if x else [])

    movies_cleaned_df['genres'] = convert_to_list(movies_cleaned_df['genres'])
    movies_cleaned_df['keywords'] = convert_to_list(movies_cleaned_df['keywords'])
    movies_cleaned_df['cast'] = convert_to_list(movies_cleaned_df['cast'])
    movies_cleaned_df['crew'] = convert_to_list(movies_cleaned_df['crew'])

    # One-hot encoding for genres
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(movies_cleaned_df['genres']), columns=mlb.classes_, index=movies_cleaned_df.index)

    # Combine encoded genres with the original dataframe
    movies_features_df = pd.concat([movies_cleaned_df, genres_encoded], axis=1)

    # Normalize vote_average, vote_count, and sentiment scores
    scaler = MinMaxScaler()
    movies_features_df[['vote_average', 'vote_count', 'sentiment']] = scaler.fit_transform(movies_features_df[['vote_average', 'vote_count', 'sentiment']])

    return movies_features_df, genres_encoded.columns.tolist(), movies_cleaned_df

movies_features_df, genre_columns, movies_cleaned_df = preprocess_data(movies_df)

# Create the recommendation function
def get_recommendations(title, similarity_matrix, movies_df):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices]

# Create feature matrix and calculate cosine similarity
features = genre_columns + ['vote_average', 'vote_count', 'sentiment']
feature_matrix = movies_features_df[features].values
similarity_matrix = cosine_similarity(feature_matrix)

# Fetch movie details from TMDB
def fetch_movie_details(movie_title):
    search = movie.search(movie_title)
    if search:
        movie_id = search[0].id
        movie_details = movie.details(movie_id)
        poster_path = f"https://image.tmdb.org/t/p/w500{movie_details.poster_path}"
        rating = movie_details.vote_average
        return poster_path, rating
    return None, None

# Streamlit interface
st.title('Movie Recommendation System')

# Input for movie title
movie_title = st.selectbox('Select a movie to get recommendations:', movies_df['title'].values)

if st.button('Get Recommendations'):
    recommendations = get_recommendations(movie_title, similarity_matrix, movies_cleaned_df)
    st.write('Here are some movies you might like:')
    for index, row in recommendations.iterrows():
        poster_url, rating = fetch_movie_details(row['title'])
        if poster_url:
            st.image(poster_url, width=100)
        st.write(f"**{row['title']}**")
        st.write(f"Rating: {row['vote_average']} | TMDB Rating: {rating if rating else 'N/A'}")
        st.write(row['overview'])
