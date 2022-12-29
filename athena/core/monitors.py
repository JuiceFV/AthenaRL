import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from athena.core.tracker import Aggregator, Tracker

logger = logging.getLogger(__name__)


class CompositeMonitor(Tracker):
    def __init__(self, trackers: Iterable[Tracker]) -> None:
        self.trackers: Dict[str, List[Tracker]] = {}
        for tracker in trackers:
            fields = tracker.get_trackable_fields()
            for field in fields:
                self.trackers.setdefault(field, []).append(tracker)
        super().__init__(list(self.trackers))

    def update(self, field: str, value: Any):
        for tracker in self.trackers[field]:
            tracker.update(field, value)


class IntervalAggMonitor(Tracker):
    def __init__(
        self,
        interval: Optional[int],
        aggregator: Aggregator,
        on_epoch_end: bool = True
    ) -> None:
        self.field = aggregator.field
        fields = ["epoch_end"] if on_epoch_end else []
        fields.append(self.field)
        super().__init__(fields)
        self.iter = 0
        self.interval = interval
        self.agg_values: List[Any] = []
        self.aggregator = aggregator

    def update(self, field: str, value: Any) -> None:
        if field == "epoch_end":
            self.flush()
            return None
        self.agg_values.append(value)
        self.iter += 1
        if self.interval and self.iter % self.interval == 0:
            logger.info(
                "Aggregating values over the recent interval for %s at iteration %s; aggregator: %s",
                self.field,
                self.iter,
                self.aggregator.__class__.__name__,
            )
            self.aggregator(self.field, self.agg_values)
            self.agg_values = []

    def flush(self) -> None:
        self.iter = 0
        if self.agg_values:
            self.aggregator(self.field, self.agg_values)
        self.agg_values = []
        self.aggregator.flush()


class ValueListMonitor(Tracker):
    def __init__(self, field: str) -> None:
        super().__init__([field])
        self.trackable_field = field
        self.values: List[Any] = []

    def update(self, field: str, value: Any) -> None:
        self.values.append(value)

    def reset(self):
        self.values.clear()


class OnEpochEndMonitor(Tracker):
    def __init__(self, callback: Callable[[Any], None], field: str = "epoch_end") -> None:
        super().__init__([field])
        self.callback = callback

    def update(self, field: str, value: Any) -> None:
        self.callback(value)
