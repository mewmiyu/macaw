import os
import shutil

from huggingface_hub import hf_hub_download


class WeightsLoader:

    def __init__(self, model_checkpoint) -> None:
        self.model_checkpoint = model_checkpoint
          
    def __call__(self):
        if not os.path.isfile(self.model_checkpoint):
            path_to_model = hf_hub_download(repo_id="Hoebelt/macaw", filename=self.model_checkpoint)
            shutil.move(path_to_model, os.getcwd())