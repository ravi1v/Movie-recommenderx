import React, { useState, useEffect } from 'react';
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
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('trending');

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    const fetchData = async () => {
      try {
        setLoading(true);
        const [trendingRes, recRes, favRes, historyRes, genresRes] = await Promise.all([
          moviesAPI.trending(20),
          moviesAPI.recommend(undefined, 20),
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
  }, [user, navigate]);

  useEffect(() => {
    if (selectedGenre) {
      const fetchGenreMovies = async () => {
        try {
          const response = await moviesAPI.getMoviesByGenre(selectedGenre, 20);
          setGenreMovies(response.data.movies);
        } catch (error) {
          console.error('Error fetching genre movies:', error);
        }
      };
      fetchGenreMovies();
    }
  }, [selectedGenre]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    try {
      const response = await moviesAPI.search(searchQuery, 20);
      setSearchResults(response.data.movies);
      setActiveTab('search');
    } catch (error) {
      console.error('Error searching:', error);
    }
  };

  const handleFavoriteChange = (movieId, isFavorite) => {
    if (isFavorite) {
      const movie = [...trending, ...recommendations, ...searchResults].find(m => m.movieId === movieId);
      if (movie) setFavorites([...favorites, movie]);
    } else {
      setFavorites(favorites.filter(m => m.movieId !== movieId));
    }
  };

  const getMoviesToDisplay = () => {
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
  };

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
          <form onSubmit={handleSearch} className="flex-1 max-w-md mx-4">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search movies..."
                className="w-full px-4 py-2 pl-10 bg-black/50 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500 transition-colors"
              />
              <svg
                className="absolute left-3 top-2.5 w-5 h-5 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
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
        {/* Genre Filter */}
        <div className="mb-6">
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

        {/* Movies Grid */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6"
        >
          {getMoviesToDisplay().map((movie) => (
            <MovieCard
              key={movie.movieId}
              movie={movie}
              isFavorite={favorites.some(f => f.movieId === movie.movieId)}
              onFavoriteChange={handleFavoriteChange}
            />
          ))}
        </motion.div>

        {getMoviesToDisplay().length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <p className="text-xl">No movies found</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

