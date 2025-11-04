# 🚀 Quick Start Guide

## Step 1: Start the Backend Server

1. **Open a terminal/PowerShell** and navigate to the server directory:
   ```powershell
   cd C:\Users\manoj\Downloads\Movie-recommendation-system-main\server
   ```

2. **Start the Flask server:**
   ```powershell
   python app.py
   ```

   **What to expect:**
   - You'll see "Loading and preprocessing movies data..."
   - Then "Data loaded successfully!"
   - Finally: "Running on http://127.0.0.1:5000"
   - The server will keep running (don't close this window)

   **⚠️ Important:** Keep this terminal window open!

---

## Step 2: Start the Frontend

1. **Open a NEW terminal/PowerShell window** (keep the backend running)

2. **Navigate to the client directory:**
   ```powershell
   cd C:\Users\manoj\Downloads\Movie-recommendation-system-main\client
   ```

3. **Start the React app:**
   ```powershell
   npm start
   ```

   **What to expect:**
   - It will take a few seconds to compile
   - Your browser should automatically open to `http://localhost:3000`
   - If not, manually open: http://localhost:3000

---

## Step 3: Use the Application

1. **Landing Page** - You'll see the beautiful landing page with movie carousel

2. **Sign Up** - Click "Get Started" → "Sign Up"
   - Enter your name, email, and password
   - Click "Sign Up"

3. **Dashboard** - After signing up, you'll be taken to:
   - **Trending** tab - Popular movies
   - **For You** tab - Personalized recommendations
   - **Search** - Type in the search bar to find movies
   - **Genre Filter** - Select a genre from the dropdown
   - **Favorites** - Save movies you like
   - **Recently Watched** - See movies you've viewed

4. **Movie Details** - Click any movie card to see:
   - Full movie information
   - Similar movies recommendations
   - Add to favorites
   - Watch trailer link

---

## 🛑 To Stop the Application

**In both terminal windows, press:** `Ctrl + C`

Then type `y` or just press Enter to confirm.

---

## 🔧 Troubleshooting

### Backend Issues:

**"Module not found" error:**
```powershell
cd server
pip install -r requirements.txt
```

**"Port 5000 already in use":**
- Close other applications using port 5000
- Or change port in `server/app.py` line 451: `app.run(debug=True, port=5001)`

**"Data files not found":**
- Make sure `movies_finalized_dataset1.csv`, `ratings.csv`, and `links.csv` are in the parent directory (not inside `server/`)

### Frontend Issues:

**"Module not found" or build errors:**
```powershell
cd client
npm install
npm start
```

**"Can't connect to API":**
- Make sure the backend is running on port 5000
- Check browser console for errors
- Verify backend is accessible at: http://localhost:5000/api/health

---

## ✅ Quick Checklist

- [ ] Backend server is running (terminal showing "Running on http://127.0.0.1:5000")
- [ ] Frontend is running (browser open at http://localhost:3000)
- [ ] Can see the landing page
- [ ] Can sign up/login
- [ ] Can see movie recommendations
- [ ] Can search for movies
- [ ] Can view movie details

---

## 📝 Notes

- **First startup** may take 30-60 seconds as it loads movie data
- **Movie posters** may show placeholders (to enable real posters, see README.md for TMDB API setup)
- Keep **both terminals open** while using the app
- The app works **offline** (all data is local)

---

**Enjoy your Movie Recommendation System! 🎬**

