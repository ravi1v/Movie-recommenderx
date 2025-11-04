import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { userAPI } from '../services/api';

const MovieCard = ({ movie, isFavorite: initialFavorite, onFavoriteChange }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [isFavorite, setIsFavorite] = useState(initialFavorite || false);
  const [loading, setLoading] = useState(false);

  const handleFavorite = async (e) => {
    e.stopPropagation();
    if (!user) return;

    setLoading(true);
    try {
      if (isFavorite) {
        await userAPI.removeFavorite(movie.movieId);
        setIsFavorite(false);
        if (onFavoriteChange) onFavoriteChange(movie.movieId, false);
      } else {
        await userAPI.addFavorite(movie.movieId);
        setIsFavorite(true);
        if (onFavoriteChange) onFavoriteChange(movie.movieId, true);
      }
    } catch (error) {
      console.error('Error updating favorite:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      whileHover={{ scale: 1.05, zIndex: 10 }}
      className="relative group cursor-pointer"
      onClick={() => navigate(`/movie/${movie.movieId}`)}
    >
      <div className="relative h-96 rounded-lg overflow-hidden glass">
        {/* Movie Poster */}
        {movie.poster_url ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="w-full h-full object-cover"
            onError={(e) => {
              // Fallback to placeholder if image fails to load
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
        ) : null}
        {/* Placeholder - shown if no poster or image fails */}
        <div 
          className={`absolute inset-0 bg-gradient-to-br from-purple-900/50 to-pink-900/50 flex items-center justify-center ${
            movie.poster_url ? 'hidden' : ''
          }`}
        >
          <div className="text-6xl opacity-20">🎬</div>
        </div>

        {/* Favorite Button */}
        {user && (
          <button
            onClick={handleFavorite}
            disabled={loading}
            className="absolute top-4 right-4 z-20 p-2 rounded-full glass hover:bg-red-500/20 transition-all"
          >
            <motion.svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill={isFavorite ? '#ef4444' : 'none'}
              stroke={isFavorite ? '#ef4444' : 'white'}
              strokeWidth="2"
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
            >
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
            </motion.svg>
          </button>
        )}

        {/* Overlay on Hover */}
        <motion.div
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent flex flex-col justify-end p-4"
        >
          <h3 className="text-xl font-bold mb-2 line-clamp-2">{movie.title}</h3>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-yellow-400">★</span>
            <span className="text-sm">{movie.avg_rating.toFixed(1)}</span>
          </div>
          <div className="flex flex-wrap gap-1 mb-2">
            {movie.genres.slice(0, 2).map((genre, idx) => (
              <span
                key={idx}
                className="text-xs px-2 py-1 bg-purple-500/20 rounded-full"
              >
                {genre}
              </span>
            ))}
          </div>
          <p className="text-sm text-gray-300 line-clamp-2">
            Click to view details
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default MovieCard;

