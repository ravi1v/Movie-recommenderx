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

# Email validation helper function
def is_valid_email(email):
    """Validate email format using regex"""
    email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    return re.match(email_pattern, email) is not None

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
movies_df['clean_title_lower'] = movies_df['clean_title'].str.lower()

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies_df['clean_title'])

# Build fast search index (for prefix/exact matching)
print("Building search index...")
search_index = {}
# Use positional index for proper iloc access (iloc uses positional, not label-based)
for pos_idx in range(len(movies_df)):
    row = movies_df.iloc[pos_idx]
    title_lower = str(row['clean_title_lower']).strip()
    if pd.isna(title_lower) or title_lower == 'nan':
        continue
    words = title_lower.split()
    # Index by first word for fast prefix search
    if words:
        first_word = words[0]
        if first_word not in search_index:
            search_index[first_word] = []
        search_index[first_word].append(pos_idx)
    # Also index full title for exact matches
    if title_lower not in search_index:
        search_index[title_lower] = []
    search_index[title_lower].append(pos_idx)

# Search result cache
search_cache = {}
max_cache_size = 1000

print("Data loaded successfully!")

# Helper functions
def clean_title(title):
    """Clean movie title for search"""
    return re.sub("[^a-zA-Z0-9 ]", "", str(title))

def search_movies(query, limit=10):
    """Ultra-fast movie search with vectorized pandas operations"""
    try:
        if not query or not query.strip():
            return movies_df.head(limit)
        
        query_lower = clean_title(query).lower().strip()
        cache_key = f"{query_lower}_{limit}"
        
        # Check cache first (instant return)
        if cache_key in search_cache:
            indices = search_cache[cache_key]
            valid_indices = [i for i in indices if 0 <= i < len(movies_df)]
            if valid_indices:
                return movies_df.iloc[valid_indices]
        
        # Strategy 1: Fast pandas vectorized substring search (MUCH faster)
        # Use pandas .str.contains() which is vectorized and optimized (vectorized = very fast)
        mask = movies_df['clean_title_lower'].str.contains(query_lower, case=False, na=False)
        substring_results = movies_df[mask]
        
        if len(substring_results) >= limit:
            results = substring_results.head(limit)
            # Store positional indices for cache (using numpy argwhere for efficiency)
            mask_positions = np.where(mask)[0]
            valid_indices = mask_positions[:limit].tolist()
            # Cache results
            if len(search_cache) >= max_cache_size:
                oldest_key = next(iter(search_cache))
                del search_cache[oldest_key]
            search_cache[cache_key] = valid_indices
            return results
        
        # Strategy 2: Fast indexed exact/prefix match (for cases where substring doesn't work well)
        exact_matches = []
        prefix_matches = []
        
        # Check for exact title match
        if query_lower in search_index:
            exact_matches.extend(search_index[query_lower])
        
        # Check for prefix matches (words starting with query)
        words = query_lower.split()
        if words:
            first_word = words[0]
            if first_word in search_index:
                prefix_matches.extend(search_index[first_word])
        
        # Combine exact and prefix matches
        fast_matches = list(dict.fromkeys(exact_matches + prefix_matches))
        
        if len(fast_matches) >= limit:
            valid_indices = [i for i in fast_matches if 0 <= i < len(movies_df)][:limit]
            if valid_indices:
                results = movies_df.iloc[valid_indices]
                # Cache results
                if len(search_cache) >= max_cache_size:
                    oldest_key = next(iter(search_cache))
                    del search_cache[oldest_key]
                search_cache[cache_key] = valid_indices
                return results
        
        # Combine substring and indexed results
        if len(substring_results) > 0:
            results = substring_results.head(limit)
            # Get positional indices from the mask
            mask_positions = np.where(mask)[0]
            valid_indices = mask_positions[:limit].tolist()
            if len(search_cache) >= max_cache_size:
                oldest_key = next(iter(search_cache))
                del search_cache[oldest_key]
            search_cache[cache_key] = valid_indices
            return results
        
        # Strategy 3: TF-IDF similarity (only if nothing found above)
        query_clean = clean_title(query)
        query_vec = vectorizer.transform([query_clean])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Optimized: only get top results, no need to sort everything
        top_indices = np.argpartition(similarity, -limit)[-limit:]
        sorted_top_indices = top_indices[np.argsort(similarity[top_indices])[::-1]]
        
        valid_indices = [int(i) for i in sorted_top_indices if 0 <= i < len(movies_df)]
        if valid_indices:
            results = movies_df.iloc[valid_indices]
            
            # Cache TF-IDF results
            if len(search_cache) >= max_cache_size:
                oldest_key = next(iter(search_cache))
                del search_cache[oldest_key]
            search_cache[cache_key] = valid_indices
            
            return results
        
        # Fallback: return empty DataFrame
        return movies_df.iloc[[]]
        
    except Exception as e:
        print(f"Error in search_movies: {e}")
        import traceback
        traceback.print_exc()
        # Simple fallback
        try:
            query_clean = clean_title(query)
            query_vec = vectorizer.transform([query_clean])
            similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.argsort(similarity)[::-1][:limit]
            return movies_df.iloc[top_indices]
        except:
            return movies_df.head(limit)

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
    """Get movie poster URL - returns None for now"""
    # Original implementation - no poster fetching
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
    
    # Validate email format
    if not is_valid_email(email):
        return jsonify({'error': 'Please enter a valid email address'}), 400
    
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
    
    # Validate email format
    if not is_valid_email(email):
        return jsonify({'error': 'Please enter a valid email address'}), 400
    
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
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        results = search_movies(query, limit)
        
        # Check if results is empty
        if len(results) == 0:
            return jsonify({'movies': []})
        
        movies_list = [movie_to_dict(row) for _, row in results.iterrows()]
        
        return jsonify({'movies': movies_list})
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Search failed', 'movies': []}), 500

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
    """Get personalized recommendations based on user favorites"""
    movie_id = request.args.get('movie_id', type=int)
    limit = int(request.args.get('limit', 20))
    
    if movie_id:
        # Get recommendations based on a specific movie
        recommendations = find_similar_movies(movie_id, limit)
    else:
        # Get recommendations based on user's favorite movies
        user_id = get_user_id_from_token()
        
        if user_id:
            # Get user's favorite movies
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT movie_id FROM favorites WHERE user_id = ?', (user_id,))
            favorite_ids = [row[0] for row in c.fetchall()]
            conn.close()
            
            if favorite_ids:
                # Generate recommendations based on all favorites
                all_recommendations = {}
                
                # Get similar movies for each favorite (limit per favorite to avoid too many)
                per_favorite_limit = max(5, limit // max(1, len(favorite_ids)))
                
                for fav_id in favorite_ids:
                    similar = find_similar_movies(fav_id, per_favorite_limit)
                    
                    # Aggregate recommendations with scores
                    for _, row in similar.iterrows():
                        rec_id = int(row.get('movieId', 0))
                        # Skip if already in favorites
                        if rec_id in favorite_ids:
                            continue
                        
                        # Use score if available, otherwise use avg_rating
                        try:
                            if 'score' in row.index and pd.notna(row['score']):
                                score = float(row['score'])
                            else:
                                score = float(row.get('avg_rating', 0)) if pd.notna(row.get('avg_rating')) else 0.0
                        except (KeyError, ValueError):
                            score = float(row.get('avg_rating', 0)) if pd.notna(row.get('avg_rating')) else 0.0
                        
                        if rec_id in all_recommendations:
                            # If already recommended, increase score (multiple favorites suggest it)
                            all_recommendations[rec_id]['score'] += score
                            all_recommendations[rec_id]['count'] += 1
                        else:
                            all_recommendations[rec_id] = {
                                'movieId': rec_id,
                                'title': str(row.get('title', '')),
                                'genres': row.get('genres', ''),
                                'imdb_url': str(row.get('imdb_url', '')),
                                'imdbId': str(row.get('imdbId', '')),
                                'avg_rating': float(row.get('avg_rating', 0)),
                                'score': score,
                                'count': 1
                            }
                
                # Convert to DataFrame-like structure and sort by score
                if all_recommendations:
                    # Sort by score (weighted by how many favorites suggest it)
                    sorted_recs = sorted(
                        all_recommendations.values(),
                        key=lambda x: x['score'] * x['count'],
                        reverse=True
                    )[:limit]
                    
                    # Get full movie data
                    rec_ids = [r['movieId'] for r in sorted_recs]
                    recommendations = movies_df[movies_df['movieId'].isin(rec_ids)]
                    
                    # Reorder to match sorted_recs order
                    id_to_order = {rec_id: idx for idx, rec_id in enumerate(rec_ids)}
                    recommendations['_sort_order'] = recommendations['movieId'].map(id_to_order)
                    recommendations = recommendations.sort_values('_sort_order').drop('_sort_order', axis=1)
                else:
                    # Fallback: return trending movies
                    recommendations = movies_df.nlargest(limit, 'avg_rating')
            else:
                # No favorites, return trending movies
                recommendations = movies_df.nlargest(limit, 'avg_rating')
        else:
            # Not logged in, return trending movies
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
    import warnings
    import logging
    # Suppress the development server warning
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    app.run(debug=True, port=5000, use_reloader=False)

