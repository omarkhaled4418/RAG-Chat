"""Flask application entry point."""

from app.config import Config
from flask import Flask


def create_app():
    """Application factory."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config.from_object(Config)

    # Enable CORS for external tools like n8n
    from flask_cors import CORS
    CORS(app)

    # Ensure required directories exist
    import os
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)

    # Register routes
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True, port=5000)
