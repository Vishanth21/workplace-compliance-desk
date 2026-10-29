"""Standalone background queue worker process for asynchronous compliance execution (D3)."""

import asyncio
import logging
import signal

from app.queue import WORKER_COUNT, start_workers, stop_workers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")


async def main() -> None:
    logger.info("Starting compliance desk worker pool (count=%d)...", WORKER_COUNT)
    start_workers(WORKER_COUNT)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # Windows compatibility

    logger.info("Worker pool active and awaiting compliance tasks.")
    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
    finally:
        logger.info("Shutting down worker pool...")
        stop_workers()
        logger.info("Worker pool stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

