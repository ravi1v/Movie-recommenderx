import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { moviesAPI, userAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import MovieCard from './MovieCard';

const MovieDetails = () => {
  const { movieId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [movie, setMovie] = useState(null);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [isFavorite, setIsFavorite] = useState(false);
  const [loading, setLoading] = useState(true);
  const [trailerUrl, setTrailerUrl] = useState(null);

  useEffect(() => {
    const fetchMovie = async () => {
      try {
        setLoading(true);
        const response = await moviesAPI.getMovie(parseInt(movieId));
        setMovie(response.data);
        setSimilarMovies(response.data.similar_movies || []);
        
        // Check if favorite
        if (user) {
          try {
            const favResponse = await userAPI.getFavorites();
            setIsFavorite(favResponse.data.movies.some(m => m.movieId === parseInt(movieId)));
          } catch (error) {
            console.error('Error checking favorites:', error);
          }
        }

        // Try to get YouTube trailer
        const searchQuery = `${response.data.title} trailer`;
        const youtubeSearchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(searchQuery)}`;
        // In a real app, you'd use YouTube API to get the embed URL
        setTrailerUrl(youtubeSearchUrl);
      } catch (error) {
        console.error('Error fetching movie:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMovie();
  }, [movieId, user]);

  const handleFavorite = async () => {
    if (!user) {
      navigate('/login');
      return;
    }

    try {
      if (isFavorite) {
        await userAPI.removeFavorite(movie.movieId);
        setIsFavorite(false);
      } else {
        await userAPI.addFavorite(movie.movieId);
        setIsFavorite(true);
      }
    } catch (error) {
      console.error('Error updating favorite:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Movie not found</h2>
          <button
            onClick={() => navigate('/dashboard')}
            className="px-6 py-3 bg-purple-600 rounded-lg hover:bg-purple-500 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cinema-dark">
      {/* Back Button */}
      <button
        onClick={() => navigate(-1)}
        className="absolute top-4 left-4 z-10 p-3 glass rounded-lg hover:bg-black/50 transition-colors"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      {/* Hero Section */}
      <div className="relative h-[60vh] overflow-hidden">
        {/* Background Poster */}
        {movie.poster_url ? (
          <div className="absolute inset-0">
            <img
              src={movie.poster_url}
              alt={movie.title}
              className="w-full h-full object-cover opacity-20"
            />
            <div className="absolute inset-0 bg-gradient-to-br from-purple-900/50 via-pink-900/50 to-red-900/50" />
          </div>
        ) : (
          <div className="absolute inset-0 bg-gradient-to-br from-purple-900/50 via-pink-900/50 to-red-900/50" />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-cinema-dark via-cinema-dark/80 to-transparent" />
        
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="relative z-10 h-full flex items-end max-w-7xl mx-auto px-4 pb-12"
        >
          <div className="flex gap-8 w-full">
            {/* Movie Poster */}
            {movie.poster_url && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="hidden md:block flex-shrink-0"
              >
                <img
                  src={movie.poster_url}
                  alt={movie.title}
                  className="w-48 h-72 rounded-lg shadow-2xl object-cover border-2 border-white/20"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              </motion.div>
            )}
            
            <div className="flex-1 max-w-3xl">
              <h1 className="text-5xl md:text-7xl font-bold mb-4">{movie.title}</h1>
            <div className="flex items-center gap-4 mb-6">
              <div className="flex items-center gap-2">
                <span className="text-yellow-400 text-2xl">★</span>
                <span className="text-xl">{movie.avg_rating.toFixed(1)}</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {movie.genres.map((genre, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-purple-500/20 border border-purple-500/50 rounded-full text-sm"
                  >
                    {genre}
                  </span>
                ))}
              </div>
            </div>
            <div className="flex gap-4">
              {user && (
                <button
                  onClick={handleFavorite}
                  className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                    isFavorite
                      ? 'bg-red-500 hover:bg-red-600'
                      : 'glass hover:bg-black/50'
                  }`}
                >
                  {isFavorite ? '❤️ Remove from Favorites' : '🤍 Add to Favorites'}
                </button>
              )}
              {trailerUrl && (
                <a
                  href={trailerUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 rounded-lg font-semibold hover:from-red-500 hover:to-pink-500 transition-all"
                >
                  🎬 Watch Trailer
                </a>
              )}
            </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-12"
        >
          <h2 className="text-3xl font-bold mb-4">Overview</h2>
          <p className="text-gray-300 text-lg leading-relaxed max-w-3xl">
            {movie.imdb_url && (
              <a
                href={movie.imdb_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300 mb-4 inline-block"
              >
                View on IMDb →
              </a>
            )}
          </p>
        </motion.div>

        {/* Similar Movies */}
        {similarMovies.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h2 className="text-3xl font-bold mb-6">Similar Movies</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
              {similarMovies.map((similarMovie) => (
                <MovieCard key={similarMovie.movieId} movie={similarMovie} />
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default MovieDetails;

