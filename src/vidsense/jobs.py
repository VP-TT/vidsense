"""Background processing jobs for the UI.

Processing a video takes minutes. Doing it inside a Streamlit script run would block
the page, and any click would cancel it, so jobs run on a worker thread owned by the
server process and the page polls their progress. One worker means videos queue up
instead of competing for the CPU and GPU.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .pipeline import Placement, process_video

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    title: str
    source: str
    status: str = "queued"  # queued, running, done or error
    progress: float = 0.0
    stage: str = ""
    message: str = "waiting for the previous video to finish"
    video_id: str | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vidsense-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self, source: Path, title: str, settings: Settings, *, placement: Placement = "reference", force: bool = False
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], title=title, source=str(source))
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, settings, placement, force)
        return job

    def _run(self, job: Job, settings: Settings, placement: Placement, force: bool) -> None:
        def progress(fraction: float, stage: str, message: str) -> None:
            job.progress, job.stage, job.message = fraction, stage, message

        job.status, job.message = "running", "starting"
        try:
            record = process_video(job.source, settings, title=job.title, placement=placement, force=force, progress=progress)
        except Exception as exc:  # shown in the UI; the full traceback goes to the server log
            log.exception("processing %s failed", job.source)
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            return
        job.video_id, job.progress, job.status = record.video_id, 1.0, "done"

    def get(self, job_id: str | None) -> Job | None:
        return self._jobs.get(job_id) if job_id else None

    def active(self) -> list[Job]:
        with self._lock:
            return sorted((j for j in self._jobs.values() if j.active), key=lambda j: j.created)
