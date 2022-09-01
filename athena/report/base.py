import abc
from typing import Dict

import torch
from athena.core.logger import LoggerMixin
from athena.core.monitors import CompositeMonitor
from athena.core.monitors import IntervalAggMonitor
from athena.core.monitors import OnEpochEndMonitor
from athena.core.monitors import ValueListMonitor
from athena.core.tracker import TrackableMixin, Tracker


class ReporterBase(CompositeMonitor, LoggerMixin):
    def __init__(
        self,
        value_list_monitors: Dict[str, ValueListMonitor],
        aggregating_monitors: Dict[str, IntervalAggMonitor],
    ) -> None:
        super().__init__()
        on_epoch_end_monitor = OnEpochEndMonitor(self.flush)
        self._value_list_monitors = value_list_monitors
        self._aggregating_monitors = aggregating_monitors
        monitors = list(value_list_monitors.values()) + list(aggregating_monitors.values()) + [on_epoch_end_monitor]
        super().__init__(monitors)
        self._reporter_trackable = _ReporterTrackable(self)

    def flush(self, epoch: int):
        self.info(f"Epoch {epoch} ended")

        for monitor in self._aggregating_monitors.values():
            monitor.flush()

    def log(self, **kwargs) -> None:
        self._reporter_trackable.notify_trackers(**kwargs)

    def __getattr__(self, field: str) -> Tracker:
        val = self._value_list_monitors.get(field, None)
        if val is None:
            val = self._aggregating_monitors.get(field, None)
        if val is None:
            raise AttributeError(f"No such field {field}.")
        return val

    @abc.abstractmethod
    def training_report(self):
        pass


class _ReporterTrackable(TrackableMixin):
    def __init__(self, reporter: ReporterBase) -> None:
        self._reporter = reporter
        super().__init__()
        self.add_tracker(reporter)

    @property
    def _trackable_val_types(self) -> Dict[str, torch.Tensor]:
        # TODO: make lazy_property
        return {field: torch.Tensor for field in self._reporter.get_trackable_fields()}
