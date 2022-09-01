import os

from athena.core.dataclasses import dataclass
from athena.core.logger import LoggerMixin
from athena.core.dtypes import TrainingOutput
from athena.core.dtypes.results import NoPublishingResults
from athena.model_managers.manager import ModelManager
from athena.publisher.publisher_base import ModelPublisher
from tinydb import Query, TinyDB


KEY_FIELD = "model_config"
VALUE_FIELD = "torchscript_path"


@dataclass
class LFSPublisher(ModelPublisher, LoggerMixin):

    publishing_file: str = "/tmp/lfs_publisher"

    def __post_init_post_parse__(self):
        self.publishing_file = os.path.abspath(self.publishing_file)
        self.db = TinyDB(self.publishing_file)
        self.info(f"Using TinyDB at {self.publishing_file}")

    def get_latest_publishing_model(self, model_manager: ModelManager, module_name: str) -> str:
        model = Query()
        key = f"{module_name}_{str({model_manager})}"
        results = self.db.search(model[KEY_FIELD] == key)
        if len(results) != 1:
            if len(results) == 0:
                raise ValueError(f"Publish the model {key} first!")
            else:
                raise RuntimeError(f"Got {len(results)} results for model_manager. {results}")
        return results[0][VALUE_FIELD]

    def do_publish(self, model_manager: ModelManager, training_output: TrainingOutput) -> NoPublishingResults:
        for model_name, path in training_output.output_paths.items():
            if not os.path.exists(path):
                raise FileExistsError(f"Given path ({path})  doesn't exist.")
            model = Query()
            key = f"{model_name}_{str(model_manager)}"
            results = self.db.search(model[KEY_FIELD] == key)
            if len(results) == 0:
                self.db.insert({KEY_FIELD: key, VALUE_FIELD: path})
                self.info(f"Store model {model_name} at {path}")
            else:
                if len(results) > 1:
                    raise RuntimeError(f"Got {len(results)} results for model_manager. {results}")
                self.db.update({VALUE_FIELD: path}, model[KEY_FIELD] == key)
                self.info(f"Updating already existing model {model_name} path to {path}")
        return NoPublishingResults(success=True)
