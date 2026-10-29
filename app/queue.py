"""Process-local queue and worker pool for asynchronous compliance execution.

Requirements (A4):
- Queue between submission and execution
- Fixed number of workers (WORKER_COUNT = 2)
- Queue carries run IDs, not state (state is kept in runs store)
- Exposes queue depth for observability
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from contracts import RunRequest, RunStatus

from .store import metrics_counters, runs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Bounded process-local worker pool (visible constant)
WORKER_COUNT: int = 2

# Queue carrying run identifiers
run_queue: asyncio.Queue[str] = asyncio.Queue()

# Request and correlation mappings for runs
run_requests: dict[str, RunRequest] = {}
run_correlations: dict[str, str] = {}

# Active background worker tasks
_workers: list[asyncio.Task] = []


async def queue_worker(worker_id: int) -> None:
    """Worker task that consumes run ids from the queue and executes the pipeline."""
    from .pipeline import run_pipeline

    logger.info("Worker %d started", worker_id)
    while True:
        try:
            run_id = await run_queue.get()
        except asyncio.CancelledError:
            break

        try:
            request = run_requests.get(run_id)
            if request is not None and run_id in runs:
                await run_pipeline(run_id, request)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Worker %d encountered error on run %s: %s", worker_id, run_id, exc)
            if run_id in runs:
                runs[run_id].status = RunStatus.FAILED
                runs[run_id].error = str(exc)
                metrics_counters["failed"] += 1
        finally:
            run_queue.task_done()
            run_requests.pop(run_id, None)


def start_workers(worker_count: int = WORKER_COUNT) -> list[asyncio.Task]:
    """Ensure background worker pool is active."""
    global _workers
    _workers = [t for t in _workers if not t.done()]
    while len(_workers) < worker_count:
        w_id = len(_workers)
        task = asyncio.create_task(queue_worker(w_id), name=f"compliance-worker-{w_id}")
        _workers.append(task)
    return _workers


async def stop_workers() -> None:
    """Stop all background workers gracefully."""
    global _workers
    for task in _workers:
        task.cancel()
    if _workers:
        await asyncio.gather(*_workers, return_exceptions=True)
    _workers.clear()


def get_queue_depth() -> int:
    """Return the number of runs currently waiting in the queue."""
    return run_queue.qsize()


def reset_queue() -> None:
    """Reset queue and request mappings for test isolation."""
    while not run_queue.empty():
        try:
            run_queue.get_nowait()
            run_queue.task_done()
        except (asyncio.QueueEmpty, ValueError):
            break
    run_requests.clear()
    run_correlations.clear()
