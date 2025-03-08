from flask import Flask
from app.routes import init_routes # We'll define this function in routes.py

app = Flask(__name__, template_folder='app/templates')
app.secret_key = "your-secret-key"  # Set a secret key for session management (use env variable in production)

# Initialize routes
init_routes(app)

if __name__ == '__main__':
    app.run(debug=True)