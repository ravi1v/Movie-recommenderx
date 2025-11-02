from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import bcrypt
import jwt
import secrets
#hello
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Secret key for JWT tokens
SECRET_KEY = secrets.token_hex(32)
app.config['SECRET_KEY'] = SECRET_KEY

# Database setup
DB_PATH = 'users.db'

def init_db():
    """Initialize SQLite database for user authentication"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  name TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # User favorites table
    c.execute('''CREATE TABLE IF NOT EXISTS favorites
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  movie_id INTEGER NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id),
                  UNIQUE(user_id, movie_id))''')
    
    # Recently watched table
    c.execute('''CREATE TABLE IF NOT EXISTS recently_watched
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  movie_id INTEGER NOT NULL,
                  watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR))

movies_df = pd.read_csv(os.path.join(DATA_DIR, 'movies_finalized_dataset1.csv'))
ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
links_df = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))

# Merge tmdbId from links to movies
movies_df = movies_df.merge(
    links_df[['movieId', 'tmdbId']], 
    on='movieId', 
    how='left'
)

# Preprocess movies data
print("Loading and preprocessing movies data...")
movies_df['clean_title'] = movies_df['title'].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", str(x)))

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies_df['clean_title'])

print("Data loaded successfully!")

# Helper functions
def clean_title(title):
    """Clean movie title for search"""
    return re.sub("[^a-zA-Z0-9 ]", "", str(title))

def search_movies(query, limit=10):
    """Search movies by title using TF-IDF"""
    query_clean = clean_title(query)
    query_vec = vectorizer.transform([query_clean])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = np.argpartition(similarity, -limit)[-limit:]
    # Sort by similarity score
    indices = indices[np.argsort(similarity[indices])][::-1]
    results = movies_df.iloc[indices]
    return results.head(limit)

def find_similar_movies(movie_id, limit=10):
    """Find similar movies using collaborative filtering"""
    try:
        # Find users who rated this movie highly (>4)
        similar_users = ratings_df[
            (ratings_df['movieId'] == movie_id) & 
            (ratings_df['rating'] > 4)
        ]['userId'].unique()
        
        if len(similar_users) == 0:
            # Fallback: return popular movies in same genre
            movie = movies_df[movies_df['movieId'] == movie_id]
            if len(movie) == 0:
                return pd.DataFrame()
            genres = movie.iloc[0]['genres'].split('|')
            genre_movies = movies_df[movies_df['genres'].str.contains('|'.join(genres), na=False)]
            return genre_movies.nlargest(limit, 'avg_rating')
        
        # Find movies liked by similar users
        similar_user_recs = ratings_df[
            (ratings_df['userId'].isin(similar_users)) & 
            (ratings_df['rating'] > 4)
        ]['movieId']
        
        if len(similar_user_recs) == 0:
            return pd.DataFrame()
        
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
        similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
        
        # Compare with all users
        all_users = ratings_df[
            (ratings_df['movieId'].isin(similar_user_recs.index)) & 
            (ratings_df['rating'] > 4)
        ]
        
        if len(all_users) == 0:
            return pd.DataFrame()
        
        all_user_recs = all_users['movieId'].value_counts() / len(all_users['userId'].unique())
        
        # Calculate recommendation score
        rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
        rec_percentages.columns = ['similar', 'all']
        rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']
        rec_percentages = rec_percentages.sort_values('score', ascending=False)
        
        # Merge with movie data
        recommendations = rec_percentages.head(limit).merge(
            movies_df, 
            left_index=True, 
            right_on='movieId'
        )
        
        return recommendations[['movieId', 'title', 'genres', 'imdb_url', 'avg_rating', 'score']]
    except Exception as e:
        print(f"Error finding similar movies: {e}")
        return pd.DataFrame()

def get_token_from_header():
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            return auth_header.split(' ')[1]  # Bearer TOKEN
        except:
            return None
    return None

def get_user_id_from_token():
    """Get user ID from JWT token"""
    token = get_token_from_header()
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload.get('user_id')
    except:
        return None

def get_poster_url(row):
    """Get movie poster URL - attempts multiple strategies"""
    # Strategy 1: Try to get poster from TMDB using tmdbId
    # Note: TMDB requires poster_path from API, not just movie ID
    # For now, we'll try constructing a potential URL pattern
    tmdb_id = row.get('tmdbId')
    if pd.notna(tmdb_id) and tmdb_id != '':
        try:
            tmdb_id_int = int(float(tmdb_id))
            # Return tmdbId so frontend can potentially use it
            # In production, you'd call TMDB API to get actual poster_path
            # For demo: return None and let frontend show placeholder
            pass
        except:
            pass
    
    # Strategy 2: Use IMDb ID (format: tt0114709)
    # Some services can construct poster URLs from IMDb IDs
    imdb_id = row.get('imdbId', '')
    if pd.notna(imdb_id) and imdb_id != '':
        try:
            # Format IMDb ID
            imdb_str = str(int(imdb_id)) if isinstance(imdb_id, (int, float)) else str(imdb_id).replace('tt', '').strip()
            imdb_formatted = f"tt{imdb_str.zfill(7)}"
            
            # Return IMDb ID formatted - frontend can try to fetch from services
            # Or you can integrate with a poster API service here
            # For now, return None - frontend will show placeholder
            # To enable posters: integrate with TMDB API (free, requires registration)
            # API endpoint: https://api.themoviedb.org/3/find/{imdb_formatted}?api_key=KEY&external_source=imdb_id
            return None
        except Exception as e:
            print(f"Error formatting IMDb ID: {e}")
    
    # No poster available - frontend will show placeholder
    return None

def movie_to_dict(row):
    """Convert movie DataFrame row to dictionary"""
    try:
        genres = []
        if pd.notna(row.get('genres', '')):
            if isinstance(row['genres'], str):
                genres = row['genres'].split('|')
            else:
                genres = []
        
        return {
            'movieId': int(row['movieId']),
            'title': str(row.get('title', 'Unknown')),
            'genres': genres,
            'imdb_url': str(row.get('imdb_url', '')),
            'imdbId': str(row.get('imdbId', '')),
            'avg_rating': float(row.get('avg_rating', 0)) if pd.notna(row.get('avg_rating')) else 0.0,
            'poster_url': get_poster_url(row)
        }
    except Exception as e:
        print(f"Error converting movie to dict: {e}")
        return {
            'movieId': 0,
            'title': 'Unknown',
            'genres': [],
            'imdb_url': '',
            'imdbId': '',
            'avg_rating': 0.0,
            'poster_url': None
        }

# API Routes

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    name = data.get('name', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if user exists
    c.execute('SELECT id FROM users WHERE email = ?', (email,))
    if c.fetchone():
        conn.close()
        return jsonify({'error': 'User already exists'}), 400
    
    # Hash password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Insert user
    c.execute('INSERT INTO users (email, password, name) VALUES (?, ?, ?)',
              (email, hashed.decode('utf-8'), name))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Generate JWT token
    token = jwt.encode({'user_id': user_id, 'email': email}, SECRET_KEY, algorithm='HS256')
    
    return jsonify({
        'token': token,
        'user': {
            'id': user_id,
            'email': email,
            'name': name
        }
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT id, email, password, name FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    user_id, user_email, hashed_password, name = user
    
    # Verify password
    if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Generate JWT token
    token = jwt.encode({'user_id': user_id, 'email': user_email}, SECRET_KEY, algorithm='HS256')
    
    return jsonify({
        'token': token,
        'user': {
            'id': user_id,
            'email': user_email,
            'name': name
        }
    })

@app.route('/api/search', methods=['GET'])
def search():
    """Search movies by title"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    results = search_movies(query, limit)
    
    movies_list = [movie_to_dict(row) for _, row in results.iterrows()]
    
    return jsonify({'movies': movies_list})

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Get movie details by ID"""
    movie = movies_df[movies_df['movieId'] == movie_id]
    
    if len(movie) == 0:
        return jsonify({'error': 'Movie not found'}), 404
    
    movie_dict = movie_to_dict(movie.iloc[0])
    
    # Get similar movies
    similar = find_similar_movies(movie_id, limit=10)
    similar_list = []
    if len(similar) > 0:
        for _, row in similar.iterrows():
            if int(row.get('movieId', 0)) != movie_id:
                similar_list.append(movie_to_dict(row))
                if len(similar_list) >= 6:
                    break
    
    movie_dict['similar_movies'] = similar_list
    
    # Track recently watched if user is logged in
    user_id = get_user_id_from_token()
    if user_id:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO recently_watched (user_id, movie_id, watched_at)
                     VALUES (?, ?, ?)''', (user_id, movie_id, datetime.now()))
        conn.commit()
        conn.close()
    
    return jsonify(movie_dict)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    """Get personalized recommendations"""
    movie_id = request.args.get('movie_id', type=int)
    limit = int(request.args.get('limit', 20))
    
    if movie_id:
        # Get recommendations based on a specific movie
        recommendations = find_similar_movies(movie_id, limit)
    else:
        # Get trending/popular movies
        recommendations = movies_df.nlargest(limit, 'avg_rating')
    
    movies_list = [movie_to_dict(row) for _, row in recommendations.iterrows()]
    
    return jsonify({'movies': movies_list})

@app.route('/api/trending', methods=['GET'])
def trending():
    """Get trending movies (highest rated)"""
    limit = int(request.args.get('limit', 20))
    trending_movies = movies_df.nlargest(limit, 'avg_rating')
    
    movies_list = [movie_to_dict(row) for _, row in trending_movies.iterrows()]
    
    return jsonify({'movies': movies_list})

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get list of all genres"""
    all_genres = set()
    for genres_str in movies_df['genres'].dropna():
        if isinstance(genres_str, str):
            all_genres.update(genres_str.split('|'))
    
    return jsonify({'genres': sorted(list(all_genres))})

@app.route('/api/movies/genre', methods=['GET'])
def get_movies_by_genre():
    """Get movies by genre"""
    genre = request.args.get('genre', '')
    limit = int(request.args.get('limit', 20))
    
    if not genre:
        return jsonify({'error': 'Genre parameter is required'}), 400
    
    genre_movies = movies_df[movies_df['genres'].str.contains(genre, na=False)]
    genre_movies = genre_movies.nlargest(limit, 'avg_rating')
    
    movies_list = [movie_to_dict(row) for _, row in genre_movies.iterrows()]
    
    return jsonify({'movies': movies_list})

@app.route('/api/user/favorites', methods=['GET'])
def get_favorites():
    """Get user's favorite movies"""
    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT movie_id FROM favorites WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    favorite_ids = [row[0] for row in c.fetchall()]
    conn.close()
    
    if not favorite_ids:
        return jsonify({'movies': []})
    
    favorite_movies = movies_df[movies_df['movieId'].isin(favorite_ids)]
    movies_list = [movie_to_dict(row) for _, row in favorite_movies.iterrows()]
    
    return jsonify({'movies': movies_list})

@app.route('/api/user/favorites', methods=['POST'])
def add_favorite():
    """Add movie to favorites"""
    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    movie_id = data.get('movie_id')
    
    if not movie_id:
        return jsonify({'error': 'movie_id is required'}), 400
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO favorites (user_id, movie_id) VALUES (?, ?)', (user_id, movie_id))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Movie added to favorites'}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Movie already in favorites'}), 400

@app.route('/api/user/favorites/<int:movie_id>', methods=['DELETE'])
def remove_favorite(movie_id):
    """Remove movie from favorites"""
    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM favorites WHERE user_id = ? AND movie_id = ?', (user_id, movie_id))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Movie removed from favorites'})

@app.route('/api/user/history', methods=['GET'])
def get_history():
    """Get user's recently watched movies"""
    user_id = get_user_id_from_token()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT DISTINCT movie_id FROM recently_watched 
                 WHERE user_id = ? ORDER BY watched_at DESC LIMIT 20''', (user_id,))
    history_ids = [row[0] for row in c.fetchall()]
    conn.close()
    
    if not history_ids:
        return jsonify({'movies': []})
    
    history_movies = movies_df[movies_df['movieId'].isin(history_ids)]
    movies_list = [movie_to_dict(row) for _, row in history_movies.iterrows()]
    
    return jsonify({'movies': movies_list})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

