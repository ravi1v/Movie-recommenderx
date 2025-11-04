# 🎬 MovieMagic - Movie Recommendation System

A fully responsive and visually stunning movie recommendation web application built with React, Flask, and machine learning.

![MovieMagic](https://img.shields.io/badge/React-18.0-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Tailwind](https://img.shields.io/badge/Tailwind-4.0-blue)

## ✨ Features

### Frontend Features
- 🎨 **Modern UI/UX** - Dark theme with glassmorphism and neon effects
- 🎭 **Smooth Animations** - Powered by Framer Motion
- 📱 **Fully Responsive** - Works seamlessly on all devices
- 🔍 **Advanced Search** - TF-IDF based movie search
- 🎯 **Personalized Recommendations** - ML-powered collaborative filtering
- ❤️ **Favorites System** - Save your favorite movies
- 📚 **Watch History** - Track recently viewed movies
- 🎬 **Genre Filtering** - Filter movies by genre
- 🌟 **Trending Movies** - Discover what's popular

### Backend Features
- 🔐 **User Authentication** - Secure JWT-based authentication
- 🤖 **ML Recommendations** - Content-based and collaborative filtering
- 📊 **SQLite Database** - User data and preferences storage
- 🔄 **RESTful API** - Clean API endpoints
- 📈 **Analytics Ready** - User behavior tracking

## 🛠️ Tech Stack

### Frontend
- **React.js** - UI framework
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **React Router** - Navigation
- **Axios** - HTTP client

### Backend
- **Flask** - Python web framework
- **SQLite** - Database
- **Scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **JWT** - Authentication
- **bcrypt** - Password hashing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Navigate to server directory:**
   ```bash
   cd server
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data files are in parent directory:**
   - `movies_finalized_dataset1.csv`
   - `ratings.csv`
   - `links.csv` (optional)

5. **Run Flask server:**
   ```bash
   python app.py
   ```
   Server will run on `http://localhost:5000`

### Frontend Setup

1. **Navigate to client directory:**
   ```bash
   cd client
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm start
   ```
   App will run on `http://localhost:3000`

## 📁 Project Structure

```
Movie-recommendation-system-main/
│
├── server/                  # Flask backend
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   └── users.db           # SQLite database (created automatically)
│
├── client/                  # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── LandingPage.js
│   │   │   ├── LoginPage.js
│   │   │   ├── SignupPage.js
│   │   │   ├── Dashboard.js
│   │   │   ├── MovieCard.js
│   │   │   └── MovieDetails.js
│   │   ├── context/        # React contexts
│   │   │   └── AuthContext.js
│   │   ├── services/       # API services
│   │   │   └── api.js
│   │   ├── App.js          # Main app component
│   │   └── index.js        # Entry point
│   └── package.json
│
├── movies_finalized_dataset1.csv  # Processed movie data
├── ratings.csv                    # User ratings
├── links.csv                      # Movie links
└── README.md
```

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### Movies
- `GET /api/search?q={query}` - Search movies
- `GET /api/movie/{movieId}` - Get movie details
- `GET /api/recommend?movie_id={id}` - Get recommendations
- `GET /api/trending` - Get trending movies
- `GET /api/genres` - Get all genres
- `GET /api/movies/genre?genre={name}` - Get movies by genre

### User
- `GET /api/user/favorites` - Get user favorites
- `POST /api/user/favorites` - Add to favorites
- `DELETE /api/user/favorites/{movieId}` - Remove from favorites
- `GET /api/user/history` - Get watch history

## 🎨 Design Features

- **Dark Theme** - Cinematic dark interface
- **Glassmorphism** - Frosted glass effects
- **Neon Glows** - Subtle purple/pink neon highlights
- **Smooth Transitions** - Framer Motion animations
- **Gradient Backgrounds** - Dynamic gradient overlays
- **Responsive Grid** - Adaptive movie card layouts

## 📊 Machine Learning

The recommendation system uses:
1. **Content-Based Filtering** - TF-IDF vectorization for movie similarity
2. **Collaborative Filtering** - User-based recommendations using rating patterns
3. **Hybrid Approach** - Combines both methods for better results

## 🔒 Security

- JWT token-based authentication
- Bcrypt password hashing
- SQL injection protection (parameterized queries)
- CORS configuration for API security

## 🌟 Usage

1. **Landing Page**: Browse featured movies or sign up
2. **Authentication**: Create account or sign in
3. **Dashboard**: 
   - View trending movies
   - Get personalized recommendations
   - Search movies
   - Filter by genre
4. **Movie Details**: View details, similar movies, and trailers
5. **Favorites**: Save movies to your favorites list

## 🚧 Future Enhancements

- [ ] Light/Dark mode toggle
- [ ] User ratings and reviews
- [ ] Movie watchlists
- [ ] Social features (share recommendations)
- [ ] Advanced filtering options
- [ ] Export recommendations
- [ ] Mobile app version

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Made with ❤️ using React, Flask, and Machine Learning
