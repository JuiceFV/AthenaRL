import itertools

import athena.core.aggregators as agg
from athena.core.monitors import IntervalAggMonitor
from athena.report import ReporterBase


class RLRankingReporter(ReporterBase):
    def __init__(self, report_interval: int) -> None:
        self.report_interval = report_interval
        super().__init__(self.value_list_monitors, self.aggregating_monitors)

    @property
    def value_list_monitors(self):
        # TODO: replace with lazy_property
        return {}

    @property
    def aggregating_monitors(self):
        return {
            name: IntervalAggMonitor(self.report_interval, aggregator)
            for name, aggregator in itertools.chain(
                [
                    (field, agg.MeanAggregator(log_field))
                    for field, log_field in [
                        ("train_ips_score", "train_ips_score"),
                        ("train_blured_ips_score", "train_blured_ips_score"),
                        ("train_baseline_loss", "train_baseline_loss")
                    ]
                ],
                [

                ]
            )
        }
