import threading
from slack_bolt import App
from src.retrieval.query_engine import query
from src.bot.responses import format_response
import logging

logger = logging.getLogger(__name__)

def run_query_in_background(client, channel, ts, question, thread_ts=None):
    """Execute RAG query in background thread and update message when done."""
    def _do_query():
        try:
            result = query(question)
            response_text = format_response(result)

            update_kwargs = {
                "channel": channel,
                "ts": ts,
                "text": response_text
            }
            client.chat_update(**update_kwargs)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            client.chat_update(
                channel=channel,
                ts=ts,
                text=f":x: Sorry, I encountered an error: {str(e)[:100]}"
            )

    thread = threading.Thread(target=_do_query, daemon=True)
    thread.start()

def register_handlers(app: App):
    @app.event("app_mention")
    def handle_mention(event, client, logger):
        """Handle @mentions in channels."""
        channel = event["channel"]
        thread_ts = event.get("thread_ts", event["ts"])
        user_text = event["text"]

        # Remove bot mention from text
        import re
        question = re.sub(r'<@[A-Z0-9]+>\s*', '', user_text).strip()

        if not question:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text="Please ask me a question! Example: @bot how do I configure X?"
            )
            return

        # Post "thinking" message IMMEDIATELY
        thinking_msg = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=":thinking_face: Searching knowledge base..."
        )

        # Run query in background thread
        run_query_in_background(
            client=client,
            channel=channel,
            ts=thinking_msg["ts"],
            question=question,
            thread_ts=thread_ts
        )

    @app.event("message")
    def handle_dm(event, client, logger):
        """Handle direct messages."""
        # Only handle DMs, not channel messages
        if event.get("channel_type") != "im":
            return

        # Ignore bot messages and message edits
        if event.get("subtype"):
            return

        channel = event["channel"]
        question = event.get("text", "").strip()

        if not question:
            return

        # Post "thinking" message IMMEDIATELY
        thinking_msg = client.chat_postMessage(
            channel=channel,
            text=":thinking_face: Searching knowledge base..."
        )

        # Run query in background thread
        run_query_in_background(
            client=client,
            channel=channel,
            ts=thinking_msg["ts"],
            question=question
        )
