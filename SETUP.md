# 🚀 Quick Setup Guide

## Step 1: Backend Setup

1. Open a terminal and navigate to the project root:
   ```bash
   cd Movie-recommendation-system-main
   ```

2. Navigate to server directory:
   ```bash
   cd server
   ```

3. Create and activate virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Make sure data files are in the parent directory (one level up from server):
   - `movies_finalized_dataset1.csv`
   - `ratings.csv`
   - `links.csv`

6. Run the Flask server:
   ```bash
   python app.py
   ```
   
   You should see: "Data loaded successfully!" and "Running on http://127.0.0.1:5000"

## Step 2: Frontend Setup

1. Open a **new terminal** (keep the backend running)

2. Navigate to client directory:
   ```bash
   cd client
   ```

3. Install Node.js dependencies:
   ```bash
   npm install
   ```

4. Start the React development server:
   ```bash
   npm start
   ```
   
   The app will automatically open at `http://localhost:3000`

## Step 3: Using the Application

1. **Landing Page**: You'll see the landing page with featured movies
2. **Sign Up**: Click "Get Started" → "Sign Up" to create an account
3. **Login**: Use your credentials to sign in
4. **Dashboard**: Browse movies, search, filter by genre
5. **Movie Details**: Click any movie card to see details and recommendations

## Troubleshooting

### Backend Issues

- **Import errors**: Make sure all packages are installed with `pip install -r requirements.txt`
- **Data file not found**: Ensure CSV files are in the parent directory (not in server/)
- **Port 5000 already in use**: Change port in `app.py` line: `app.run(debug=True, port=5001)`

### Frontend Issues

- **Module not found**: Run `npm install` again
- **API connection errors**: Make sure backend is running on port 5000
- **Build errors**: Clear node_modules and reinstall: `rm -rf node_modules && npm install`

### Database Issues

- If `users.db` has issues, delete it and restart the server (it will be recreated)

## Production Build

### Frontend
```bash
cd client
npm run build
```

### Backend
For production, use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn app:app
```

## Environment Variables (Optional)

Create `.env` file in `client/` directory:
```
REACT_APP_API_URL=http://localhost:5000/api
```

## Notes

- Backend takes a few seconds to load data on first startup
- Database is created automatically on first run
- All user data is stored locally in SQLite

---

Happy coding! 🎬

