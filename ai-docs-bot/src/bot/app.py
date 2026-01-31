from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.config.settings import get_settings
from src.bot.handlers import register_handlers
import logging

logging.basicConfig(level=logging.INFO)

def create_app() -> App:
    settings = get_settings()
    app = App(token=settings.SLACK_BOT_TOKEN)
    register_handlers(app)
    
    @app.error
    def global_error_handler(error, body, logger):
        logger.error(f"Unhandled error: {error}")
        logger.debug(f"Request body: {body}")
    
    return app

def run_bot():
    settings = get_settings()
    app = create_app()
    handler = SocketModeHandler(app, settings.SLACK_APP_TOKEN)
    handler.start()

if __name__ == "__main__":
    run_bot()
