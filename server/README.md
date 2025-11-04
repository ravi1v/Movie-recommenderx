# Backend Server - Flask API

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data files are in parent directory:**
   - The server expects `movies_finalized_dataset1.csv` and `ratings.csv` in the parent directory
   - These files should be copied from the root project directory

3. **Run the server:**
   ```bash
   python app.py
   ```

## Database

The SQLite database (`users.db`) is created automatically on first run with the following tables:
- `users` - User accounts
- `favorites` - User favorite movies
- `recently_watched` - User watch history

## API Configuration

Default configuration:
- Host: `localhost`
- Port: `5000`
- Debug: `True` (development mode)

To change configuration, modify `app.py` or use environment variables.

## Notes

- The server loads movie data into memory on startup (may take a few seconds)
- JWT tokens are generated with a random secret key each time (in production, use a fixed key)
- CORS is enabled for development (configure for production)

