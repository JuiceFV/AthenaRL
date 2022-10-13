import functools
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from athena.core.logger import LoggerMixin


class Tracker:
    def __init__(self, trackable_fields: List[str]) -> None:
        super().__init__()
        if not isinstance(trackable_fields, list):
            raise TypeError("Trackable fields must be passed as list.")
        self.trackable_fields = trackable_fields

    def get_trackable_fields(self) -> List[str]:
        return self.trackable_fields

    def update(self, field: str, value: Any) -> None:
        pass


class Aggregator(LoggerMixin):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def __call__(self, field: str, values) -> None:
        if field != self.field:
            raise ValueError(f"Got {field}; expected {self.field}")
        self.aggregate(values)

    def aggregate(self, values):
        pass

    def flush(self):
        pass


class TrackableMixin(LoggerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._trackers: Dict[type, List[Tracker]] = {val: [] for val in self._trackable_val_types}

    @property
    def _trackable_val_types(self) -> Dict[str, type]:
        raise NotImplementedError

    def add_tracker(self, tracker: Tracker) -> "TrackableMixin":
        trackable_fields = tracker.get_trackable_fields()
        unkonwn_fields = [field for field in trackable_fields if field not in self._trackable_val_types]
        if unkonwn_fields:
            self.warning(f"{unkonwn_fields} are non-trackable for {type(self)}")
        for field in trackable_fields:
            if field in self._trackers and tracker not in self._trackers[field]:
                self._trackers[field].append(tracker)
        return self

    def add_trackers(self, trackers: List[Tracker]) -> "TrackableMixin":
        for tracker in trackers:
            self.add_tracker(tracker)
        return self

    def notify_trackers(self, **kwargs) -> None:
        for field, value in kwargs.items():
            if value is None:
                continue

            if field not in self._trackers:
                raise IndexError(f"Unkonw field: {field}")

            if self._trackable_val_types[field] == torch.Tensor:
                try:
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value)
                    if len(value.shape) == 0:
                        value = value.reshape(1)
                    value = value.detach()
                except Exception as ex:
                    self.warning(f"Exception {ex} arised but it's fine due to there is no type convention.")
                    pass

            for tracker in self._trackers[field]:
                tracker.update(field, value)


def trackable(cls: Optional[object] = None, **kwargs) -> Union[Callable[[object], TrackableMixin], TrackableMixin]:
    if not kwargs:
        raise RuntimeError("No trackable fields were passed.")
    trackable_val_types = kwargs

    def make_trackable(cls: object) -> TrackableMixin:
        if hasattr(cls, "add_tracker") or hasattr(cls, "notify_trackers"):
            raise TypeError(f"The {cls} already tracker.")

        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self, "_trackable_val_types") or hasattr(self, "_trackers"):
                raise RuntimeError("Original class shouldn't implement _trackable_val_types and _trackers")
            self._trackable_val_types = trackable_val_types
            self._trackers = {val: [] for val in trackable_val_types}

        cls.__init__ = new_init
        cls.add_tracker = TrackableMixin.add_tracker
        cls.add_trackers = TrackableMixin.add_trackers
        cls.notify_trackers = TrackableMixin.notify_trackers
        return cls

    return make_trackable if cls is None else make_trackable(cls)
