"""PythonAnywhere WSGI Configuration for PAIMANA AI (FastAPI).

Copy the contents of this file into your PythonAnywhere WSGI configuration file:
/var/www/<your-username>_pythonanywhere_com_wsgi.py
"""
import sys
from pathlib import Path

# 1. Update this path to match your project location on PythonAnywhere
PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 2. Import FastAPI app and adapt ASGI -> WSGI using a2wsgi
from app.main import app as asgi_app
from a2wsgi import ASGIMiddleware

application = ASGIMiddleware(asgi_app)
