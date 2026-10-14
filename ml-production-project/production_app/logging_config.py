
import logging

from production_app.config import LOG_LEVEL


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
