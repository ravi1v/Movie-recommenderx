import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { moviesAPI, userAPI } from '../services/api';
import MovieCard from './MovieCard';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [trending, setTrending] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [recentlyWatched, setRecentlyWatched] = useState([]);
  const [genres, setGenres] = useState([]);
  const [selectedGenre, setSelectedGenre] = useState('');
  const [genreMovies, setGenreMovies] = useState([]);
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('trending');

  // Simple cache for API responses
  const [apiCache, setApiCache] = useState(new Map());

  // Debounce hook
  const useDebounce = (value, delay) => {
    const [debouncedValue, setDebouncedValue] = useState(value);
    useEffect(() => {
      const handler = setTimeout(() => {
        setDebouncedValue(value);
      }, delay);
      return () => {
        clearTimeout(handler);
      };
    }, [value, delay]);
    return debouncedValue;
  };

  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    const fetchData = async () => {
      try {
        setLoading(true);
        const [trendingRes, recRes, favRes, historyRes, genresRes] = await Promise.all([
          moviesAPI.trending(20, selectedLanguage || null),
          moviesAPI.recommend(undefined, 20, selectedLanguage || null),
          userAPI.getFavorites(),
          userAPI.getHistory(),
          moviesAPI.getGenres(),
        ]);

        setTrending(trendingRes.data.movies);
        setRecommendations(recRes.data.movies);
        setFavorites(favRes.data.movies || []);
        setRecentlyWatched(historyRes.data.movies || []);
        setGenres(genresRes.data.genres || []);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [user, navigate, selectedLanguage]);

  const handleSearch = useCallback(async (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      if (activeTab === 'search') {
        setActiveTab('trending');
      }
      return;
    }

    // Only search if query is at least 2 characters
    if (query.trim().length < 2) {
      return;
    }

    const cacheKey = `search-${query.trim()}-${selectedLanguage || 'all'}`;
    if (apiCache.has(cacheKey)) {
      setSearchResults(apiCache.get(cacheKey));
      setActiveTab('search');
      return;
    }

    try {
      setSearchLoading(true);
      const searchResponse = await moviesAPI.search(query.trim(), 20, selectedLanguage || null);
      const searchMovies = searchResponse.data.movies || [];

      if (searchMovies.length > 0) {
        const firstMovie = searchMovies[0];
        const recResponse = await moviesAPI.recommend(firstMovie.movieId, 19, selectedLanguage || null); // Get 19 recommendations
        const recommendations = recResponse.data.movies || [];

        // Combine: all search results first, then recommendations (avoid duplicates)
        const searchIds = new Set(searchMovies.map(m => m.movieId));
        const filteredRecommendations = recommendations.filter(m => !searchIds.has(m.movieId));
        const combined = [...searchMovies, ...filteredRecommendations];
        setSearchResults(combined);
        setApiCache(prev => new Map(prev).set(cacheKey, combined));
      } else {
        setSearchResults([]);
      }
      setActiveTab('search');
    } catch (error) {
      console.error('Error searching:', error);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  }, [activeTab, selectedLanguage, apiCache]);

  // Debounced search effect
  useEffect(() => {
    if (debouncedSearchQuery) {
      handleSearch(debouncedSearchQuery);
    }
  }, [debouncedSearchQuery, handleSearch]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch(searchQuery);
    }
  };

  useEffect(() => {
    if (selectedGenre) {
      const fetchGenreMovies = async () => {
        try {
          const response = await moviesAPI.getMoviesByGenre(selectedGenre, 20, selectedLanguage || null);
          setGenreMovies(response.data.movies);
        } catch (error) {
          console.error('Error fetching genre movies:', error);
        }
      };
      fetchGenreMovies();
    }
  }, [selectedGenre, selectedLanguage]);


  const handleFavoriteChange = async (movieId, isFavorite) => {
    try {
      if (isFavorite) {
        // Add to favorites via API
        await userAPI.addFavorite(movieId);
        const movie = [...trending, ...recommendations, ...searchResults, ...genreMovies].find(m => m.movieId === movieId);
        if (movie) {
          const updatedFavorites = [...favorites, movie];
          setFavorites(updatedFavorites);
          // Refresh recommendations based on updated favorites
          const recRes = await moviesAPI.recommend(undefined, 20);
          setRecommendations(recRes.data.movies);
        } else {
          // If movie not found in current lists, fetch favorites from server
          const favRes = await userAPI.getFavorites();
          const updatedFavorites = favRes.data.movies || [];
          setFavorites(updatedFavorites);
          // Refresh recommendations
          const recRes = await moviesAPI.recommend(undefined, 20);
          setRecommendations(recRes.data.movies);
        }
      } else {
        // Remove from favorites via API
        await userAPI.removeFavorite(movieId);
        const updatedFavorites = favorites.filter(m => m.movieId !== movieId);
        setFavorites(updatedFavorites);
        // Refresh recommendations based on updated favorites
        const recRes = await moviesAPI.recommend(undefined, 20);
        setRecommendations(recRes.data.movies);
      }
    } catch (error) {
      console.error('Error updating favorite:', error);
    }
  };

  const getMoviesToDisplay = useMemo(() => {
    switch (activeTab) {
      case 'trending':
        return trending;
      case 'recommended':
        return recommendations;
      case 'favorites':
        return favorites;
      case 'history':
        return recentlyWatched;
      case 'search':
        return searchResults;
      case 'genre':
        return genreMovies;
      default:
        return trending;
    }
  }, [activeTab, trending, recommendations, favorites, recentlyWatched, searchResults, genreMovies]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-cinema-dark">
      {/* Header */}
      <header className="sticky top-0 z-50 glass backdrop-blur-lg border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <motion.h1
            className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent"
            whileHover={{ scale: 1.05 }}
          >
            MovieMagic
          </motion.h1>

          {/* Search Bar */}
          <form onSubmit={(e) => { e.preventDefault(); }} className="flex-1 max-w-md mx-4">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Search movies... (press Enter to search)"
                className="w-full px-4 py-2 pl-10 pr-10 bg-black/50 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500 transition-colors"
              />
              {searchLoading ? (
                <div className="absolute right-3 top-2.5">
                  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-purple-500"></div>
                </div>
              ) : (
                <svg
                  className="absolute left-3 top-2.5 w-5 h-5 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              )}
            </div>
          </form>

          <div className="flex items-center gap-4">
            <span className="text-gray-300">Welcome, {user?.name || user?.email}</span>
            <button
              onClick={logout}
              className="px-4 py-2 bg-red-500/20 border border-red-500/50 rounded-lg hover:bg-red-500/30 transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Filters */}
        <div className="mb-6 flex gap-4 flex-wrap">
          <div>
            <label className="block text-sm font-medium mb-2">Filter by Genre</label>
            <select
              value={selectedGenre}
              onChange={(e) => {
                setSelectedGenre(e.target.value);
                if (e.target.value) setActiveTab('genre');
              }}
              className="px-4 py-2 bg-black/50 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
            >
              <option value="">All Genres</option>
              {genres.map((genre) => (
                <option key={genre} value={genre}>
                  {genre}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Filter by Language</label>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="px-4 py-2 bg-black/50 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
            >
              <option value="">All Languages</option>
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="it">Italian</option>
              <option value="ru">Russian</option>
              <option value="ja">Japanese</option>
              <option value="ko">Korean</option>
              <option value="zh">Chinese</option>
              <option value="hi">Hindi</option>
              <option value="kn">Kannada</option>
              <option value="te">Telugu</option>
            </select>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-8 overflow-x-auto">
          {[
            { id: 'trending', label: 'Trending' },
            { id: 'recommended', label: 'For You' },
            { id: 'favorites', label: 'Favorites' },
            { id: 'history', label: 'Recently Watched' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setSelectedGenre('');
              }}
              className={`px-6 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-purple-600 to-pink-600'
                  : 'bg-black/50 border border-white/10 hover:bg-black/70'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Movies Grid - Optimized for stable layout */}
        <div
          key={activeTab}
          className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 md:gap-6"
          style={{ minHeight: '400px' }}
        >
          {(() => {
            const favoriteIds = new Set(favorites.map(f => f.movieId));
            const movies = getMoviesToDisplay;
            return movies.length > 0 ? (
              movies.map((movie) => (
                <MovieCard
                  key={movie.movieId}
                  movie={movie}
                  isFavorite={favoriteIds.has(movie.movieId)}
                  onFavoriteChange={handleFavoriteChange}
                />
              ))
            ) : (
              <div className="col-span-full text-center py-12 text-gray-400">
                <p className="text-xl">No movies found</p>
              </div>
            );
          })()}
        </div>

      </div>
    </div>
  );
};

export default Dashboard;

