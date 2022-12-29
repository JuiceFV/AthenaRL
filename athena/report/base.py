import abc
from typing import Dict

import torch

from athena import lazy_property
from athena.core.dtypes.results import TrainingReport
from athena.core.logger import LoggerMixin
from athena.core.monitors import (CompositeMonitor, IntervalAggMonitor,
                                  OnEpochEndMonitor, ValueListMonitor)
from athena.core.tracker import TrackableMixin, Tracker


class ReporterBase(CompositeMonitor, LoggerMixin):
    def __init__(
        self,
        value_list_monitors: Dict[str, ValueListMonitor],
        aggregating_monitors: Dict[str, IntervalAggMonitor],
    ) -> None:
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
            val = val.aggregator
        return val

    @abc.abstractmethod
    def training_report(self) -> TrainingReport:
        pass


class _ReporterTrackable(TrackableMixin):
    def __init__(self, reporter: ReporterBase) -> None:
        self._reporter = reporter
        super().__init__()
        self.add_tracker(reporter)

    @lazy_property
    def _trackable_val_types(self) -> Dict[str, torch.Tensor]:
        return {field: torch.Tensor for field in self._reporter.get_trackable_fields()}
