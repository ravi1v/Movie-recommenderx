import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth API
export const authAPI = {
  register: (email, password, name) => 
    api.post('/auth/register', { email, password, name }),
  login: (email, password) => 
    api.post('/auth/login', { email, password }),
};

// Movies API
export const moviesAPI = {
  search: (query, limit = 10, language = null) =>
    api.get('/search', { params: { q: query, limit, language: language || undefined } }),
  getMovie: (movieId) =>
    api.get(`/movie/${movieId}`),
  recommend: (movieId = null, limit = 20, language = null) =>
    api.get('/recommend', { params: { movie_id: movieId, limit, language: language || undefined } }),
  trending: (limit = 20, language = null) =>
    api.get('/trending', { params: { limit, language: language || undefined } }),
  getGenres: () =>
    api.get('/genres'),
  getMoviesByGenre: (genre, limit = 20, language = null) =>
    api.get('/movies/genre', { params: { genre, limit, language: language || undefined } }),
};

// User API
export const userAPI = {
  getFavorites: () =>
    api.get('/user/favorites'),
  addFavorite: (movieId) =>
    api.post('/user/favorites', { movie_id: movieId }),
  removeFavorite: (movieId) =>
    api.delete(`/user/favorites/${movieId}`),
  getHistory: () =>
    api.get('/user/history'),
  rateMovie: (movieId, rating) =>
    api.post('/user/rate', { movie_id: movieId, rating }),
  getUserRating: (movieId) =>
    api.get(`/user/rating/${movieId}`),
};

export default api;

