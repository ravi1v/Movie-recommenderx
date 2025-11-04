# Frontend Client - React App

## Setup Instructions

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure API URL (optional):**
   - Create `.env` file in `client` directory
   - Add: `REACT_APP_API_URL=http://localhost:5000/api`
   - Default is `http://localhost:5000/api`

3. **Start development server:**
   ```bash
   npm start
   ```

The app will open at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Features

- Responsive design using Tailwind CSS
- Smooth animations with Framer Motion
- Client-side routing with React Router
- Authentication context for user management
- API service layer for backend communication

## Project Structure

- `src/components/` - React components
- `src/context/` - React contexts (Auth)
- `src/services/` - API service functions
- `src/App.js` - Main app with routing
- `src/index.js` - Entry point

## Browser Support

Modern browsers (Chrome, Firefox, Safari, Edge)

