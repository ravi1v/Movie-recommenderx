import requests
import pandas as pd
import os

# Your TMDB API key
API_KEY = "69e49ef6be3824b5b98da5dda29a7954"

# Dataset file name
DATA_FILE = "movie_dataset.csv"

# Language mapping dictionary
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "it": "Italian",
    "ru": "Russian"
}

def get_language_name(code):
    """Return full language name from code"""
    return LANGUAGE_MAP.get(code, code.upper())

def fetch_movies(movie_name=None, language=None):
    """Fetch movies from TMDB API by name or language"""
    if movie_name:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    else:
        if not language:
            language = "en"
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language={language}"

    response = requests.get(url)
    data = response.json()
    movie_list = []

    if "results" in data and data["results"]:
        for movie in data["results"]:
            title = movie.get("title", "N/A")
            original_lang = movie.get("original_language", "N/A")
            release_date = movie.get("release_date", "N/A")
            overview = movie.get("overview", "N/A")
            vote_avg = movie.get("vote_average", 0)
            poster_path = movie.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

            movie_list.append({
                "Title": title,
                "Original Language": get_language_name(original_lang),
                "Release Date": release_date,
                "Rating": vote_avg,
                "Poster URL": poster_url,
                "Overview": overview
            })
    return pd.DataFrame(movie_list)
            
# -------------------------------
# MAIN PROGRAM
# -------------------------------

# Check if dataset already exists
if os.path.exists("movie_dataset.csv"):
    print(f"\n📂 Found existing dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df)} movies from dataset.")
else:
    print("\n⚙️ No existing dataset found. Fetching new data...")
    df = pd.DataFrame()

# Ask user for input
movie_name = input("\nEnter movie name (or press Enter to skip): ").strip()
language = input("Enter language code (e.g., en=English, hi=Hindi, kn=Kannada, press Enter for all): ").strip()

# Fetch new movies
new_df = fetch_movies(movie_name if movie_name else None, language if language else None)

if not new_df.empty:
    # Combine and remove duplicates
    combined = pd.concat([df, new_df]).drop_duplicates(subset=["Title"])
    combined.to_csv(DATA_FILE, index=False)
    df = combined
    print(f"\n✅ Dataset updated and saved as '{DATA_FILE}'.")
else:
    print("⚠️ No new movies found from API.")

# Show preview
print(f"\n🎬 Showing top 10 movies:\n")
print(df.head(10))
