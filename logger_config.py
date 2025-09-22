import logging
import sys
from pathlib import Path

# Create logs directory if not exists
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# Configure root logger
logger = logging.getLogger("rag_app")
logger.setLevel(logging.DEBUG)  # capture everything, handlers will filter

# ---- Console handler ----
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # show info+ in console
console_format = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(console_format)

# ---- File handler ----
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)  # log all levels to file
file_format = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_format)

# ---- Attach handlers (avoid duplicates if re-imported) ----
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Optional: prevent propagation to root logger
logger.propagate = False
