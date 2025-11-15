import React, { useState, useEffect, memo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { userAPI } from '../services/api';

// Star Rating Component
const StarRating = ({ rating, onRate, readonly = false }) => {
  const [hoverRating, setHoverRating] = useState(0);

  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={readonly}
          onClick={() => !readonly && onRate(star)}
          onMouseEnter={() => !readonly && setHoverRating(star)}
          onMouseLeave={() => !readonly && setHoverRating(0)}
          className={`text-lg ${readonly ? 'cursor-default' : 'cursor-pointer hover:scale-110'} transition-transform`}
        >
          <span
            className={`${
              star <= (hoverRating || rating)
                ? 'text-yellow-400'
                : 'text-gray-400'
            }`}
          >
            ★
          </span>
        </button>
      ))}
    </div>
  );
};

const MovieCard = memo(({ movie, isFavorite: initialFavorite, onFavoriteChange }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [isFavorite, setIsFavorite] = useState(initialFavorite || false);
  const [loading, setLoading] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [userRating, setUserRating] = useState(null);
  const [ratingLoading, setRatingLoading] = useState(false);

  useEffect(() => {
    setIsFavorite(initialFavorite || false);
  }, [initialFavorite]);

  // Fetch user's rating for this movie
  useEffect(() => {
    if (user && movie.movieId) {
      const fetchUserRating = async () => {
        try {
          const response = await userAPI.getUserRating(movie.movieId);
          setUserRating(response.data.rating || null);
        } catch (error) {
          // User hasn't rated this movie yet
          setUserRating(null);
        }
      };
      fetchUserRating();
    }
  }, [user, movie.movieId]);

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

  const handleRating = async (rating) => {
    if (!user) return;

    setRatingLoading(true);
    try {
      await userAPI.rateMovie(movie.movieId, rating);
      setUserRating(rating);
      // Optionally refresh average rating or notify parent
    } catch (error) {
      console.error('Error rating movie:', error);
    } finally {
      setRatingLoading(false);
    }
  };

  return (
    <div
      className="relative group cursor-pointer"
      onClick={() => navigate(`/movie/${movie.movieId}`)}
    >
      <div className="relative aspect-[2/3] rounded-lg overflow-hidden glass bg-black/20" style={{ contain: 'layout style paint' }}>
        {/* Movie Poster */}
        {movie.poster_url && !imageError ? (
          <>
            <img
              src={movie.poster_url}
              alt={movie.title}
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                imageLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              onLoad={() => setImageLoaded(true)}
              onError={() => {
                setImageError(true);
                setImageLoaded(false);
              }}
              loading="lazy"
            />
            {!imageLoaded && (
              <div className="absolute inset-0 bg-gradient-to-br from-purple-900/50 to-pink-900/50 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 bg-gradient-to-br from-purple-900/50 to-pink-900/50 flex items-center justify-center">
            <div className="text-6xl opacity-20">🎬</div>
          </div>
        )}

        {/* Favorite Button */}
        {user && (
          <button
            onClick={handleFavorite}
            disabled={loading}
            className="absolute top-2 right-2 z-20 p-2 rounded-full glass hover:bg-red-500/20 transition-all backdrop-blur-sm"
            aria-label={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill={isFavorite ? '#ef4444' : 'none'}
              stroke={isFavorite ? '#ef4444' : 'white'}
              strokeWidth="2"
              className="transition-transform hover:scale-110"
            >
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
            </svg>
          </button>
        )}

        {/* Overlay on Hover */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-end p-4">
          <h3 className="text-lg font-bold mb-1 line-clamp-2 text-white">{movie.title}</h3>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-yellow-400">★</span>
            <span className="text-sm text-white">{movie.avg_rating?.toFixed(1) || 'N/A'}</span>
          </div>
          {/* User Rating */}
          {user && (
            <div className="mb-2">
              <div className="text-xs text-gray-300 mb-1">Your Rating:</div>
              <StarRating
                rating={userRating}
                onRate={handleRating}
                readonly={ratingLoading}
              />
            </div>
          )}
          {movie.genres && movie.genres.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {movie.genres.slice(0, 2).map((genre, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2 py-0.5 bg-purple-500/30 rounded-full text-white"
                >
                  {genre}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

MovieCard.displayName = 'MovieCard';

export default MovieCard;

