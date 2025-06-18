import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# ---------------------------------------------------------------------------- #
#                               Logging Configuration                          #
# ---------------------------------------------------------------------------- #
# Configure logging for the entire application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
common_logger = logging.getLogger(__name__) 

# Helper functions for consistent logging
def log_info(message: str):
    common_logger.info(message)

def log_error(message: str, exc_info=False):
    common_logger.error(message, exc_info=exc_info)

# ---------------------------------------------------------------------------- #
#                             Asynchronous Utilities                           #
# ---------------------------------------------------------------------------- #

async def run_in_executor(executor: ThreadPoolExecutor, func, *args: Any) -> Any:
    """
    Helper to run a blocking function in a ThreadPoolExecutor from an async context.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)