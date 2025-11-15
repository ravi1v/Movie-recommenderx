import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { moviesAPI } from '../services/api';

const LandingPage = () => {
  const navigate = useNavigate();
  const [featuredMovies, setFeaturedMovies] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFeatured = async () => {
      try {
        const response = await moviesAPI.trending(10);
        setFeaturedMovies(response.data.movies);
      } catch (error) {
        console.error('Error fetching featured movies:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchFeatured();
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-cinema-dark via-cinema-gray to-cinema-dark">
      {/* Hero Section */}
      <motion.section
        className="relative h-screen flex items-center justify-center overflow-hidden"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-purple-900/20 via-pink-900/20 to-red-900/20" />
        
        {/* Animated Background Particles */}
        <div className="absolute inset-0">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-white/30 rounded-full"
              initial={{
                x: Math.random() * 1000,
                y: Math.random() * 800,
              }}
              animate={{
                y: [null, Math.random() * 800],
                x: [null, Math.random() * 1000],
              }}
              transition={{
                duration: Math.random() * 3 + 2,
                repeat: Infinity,
                ease: 'linear',
              }}
            />
          ))}
        </div>

        <div className="relative z-10 text-center px-4">
          <motion.h1
            className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent"
            variants={itemVariants}
          >
            MovieMagic
          </motion.h1>
          <motion.p
            className="text-xl md:text-2xl text-gray-300 mb-8 max-w-2xl mx-auto"
            variants={itemVariants}
          >
            Discover Your Next Favorite Film with AI-Powered Recommendations
          </motion.p>
          <motion.button
            className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full text-lg font-semibold hover:from-purple-500 hover:to-pink-500 transition-all duration-300 neon-glow transform hover:scale-105"
            variants={itemVariants}
            onClick={() => navigate('/login')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Get Started
          </motion.button>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          <div className="w-6 h-10 border-2 border-white/50 rounded-full flex items-start justify-center p-2">
            <motion.div
              className="w-1 h-3 bg-white rounded-full"
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          </div>
        </motion.div>
      </motion.section>

      {/* Featured Movies Carousel */}
      <section className="py-20 px-4">
        <motion.h2
          className="text-4xl font-bold text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Featured Movies
        </motion.h2>
        
        {loading ? (
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
          </div>
        ) : (
          <div className="max-w-7xl mx-auto overflow-hidden">
            <motion.div
              className="flex gap-6"
              animate={{
                x: [0, -1000],
              }}
              transition={{
                duration: 30,
                repeat: Infinity,
                ease: 'linear',
              }}
            >
              {[...featuredMovies, ...featuredMovies].map((movie, index) => (
                <motion.div
                  key={`${movie.movieId}-${index}`}
                  className="flex-shrink-0 w-64 h-96 rounded-lg overflow-hidden glass group cursor-pointer"
                  whileHover={{ scale: 1.05, zIndex: 10 }}
                  onClick={() => navigate(`/movie/${movie.movieId}`)}
                >
                  <div className="relative h-full">
                    {/* Movie Poster */}
                    {movie.poster_url ? (
                      <img
                        src={movie.poster_url}
                        alt={movie.title}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                      />
                    ) : null}
                    {/* Placeholder */}
                    <div 
                      className={`absolute inset-0 bg-gradient-to-br from-purple-900/50 to-pink-900/50 flex items-center justify-center ${
                        movie.poster_url ? 'hidden' : 'flex'
                      }`}
                    >
                      <div className="text-6xl opacity-30 mb-4">🎬</div>
                    </div>
                    {/* Hover Overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
                      <h3 className="text-lg font-bold mb-2">{movie.title}</h3>
                      <p className="text-sm text-gray-300 line-clamp-2">
                        {movie.genres?.join(', ') || 'Movie'}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-yellow-400">★</span>
                        <span className="text-sm">{movie.avg_rating?.toFixed(1) || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        )}
      </section>
    </div>
  );
};

export default LandingPage;

