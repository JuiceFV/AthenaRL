import itertools

import athena.core.aggregators as agg
from athena.core.dtypes.results import Seq2SlateTrainingReport
from athena.core.monitors import IntervalAggMonitor
from athena.report import ReporterBase


class Seq2SlateReporter(ReporterBase):
    def __init__(self, report_interval: int = 10000) -> None:
        self.report_interval = report_interval
        super().__init__(self.value_list_monitors, self.aggregating_monitors)

    @property
    def value_list_monitors(self):
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
                        ("train_baseline_loss", "train_baseline_loss"),
                        ("train_logged_slate_rank_probas", "train_logged_slate_rank_probas"),
                        ("train_ips_ratio", "train_ips_ratio"),
                        ("train_blured_ips_ratio", "train_blured_ips_ratio"),
                        ("train_advantages", "train_advantages")
                    ]
                ],
                [
                    (f"{field}_tb", agg.TensorboardHistogramAndMeanAggregator(field, log_field))
                    for field, log_field in [
                        ("train_ips_score", "train_ips_score"),
                        ("train_blured_ips_score", "train_blured_ips_score"),
                        ("train_baseline_loss", "train_baseline_loss"),
                        ("train_logged_slate_rank_probas", "train_logged_slate_rank_probas"),
                        ("train_ips_ratio", "train_ips_ratio"),
                        ("train_blured_ips_ratio", "train_blured_ips_ratio"),
                        ("train_advantages", "train_advantages")
                    ]
                ]
            )
        }

    def training_report(self) -> Seq2SlateTrainingReport:
        return Seq2SlateTrainingReport()
