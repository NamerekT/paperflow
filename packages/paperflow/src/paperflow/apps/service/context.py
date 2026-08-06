"""What the request handlers touch once the service has started.

Built once, in `composition.build`, and stashed on `app.state` by the lifespan. Kept to
exactly what a route handler needs - today, a way to enqueue a document - rather than
exposing the whole composition root: the extra reach would only invite a handler to
reach past the queue and touch the archive directly, defeating the point of the queue.
"""

from dataclasses import dataclass

from paperflow.apps.service.worker import QueueWorker


@dataclass(frozen=True, slots=True)
class ServiceContext:
    worker: QueueWorker
