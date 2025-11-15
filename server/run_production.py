"""
Production server runner using Waitress
For Windows: waitress-serve --host=127.0.0.1 --port=5000 app:app

Install waitress first:
    pip install waitress
"""
from waitress import serve
from app import app

if __name__ == '__main__':
    print("Starting production server with Waitress...")
    print("Server running on http://127.0.0.1:5000")
    serve(app, host='127.0.0.1', port=5000, threads=4)

