import os
import shutil

from huggingface_hub import hf_hub_download


class WeightsLoader:
    def __init__(self, model_checkpoint: str) -> None:
        """Initialises the Weightsloader class with the model_checkpoint name.

        Args:
            model_checkpoint (str): The name of the model_checkpoint file
        """
        self.model_checkpoint = model_checkpoint

    def __call__(self):
        """Downloads the model_checkpoint file from huggingface.

        Raises:
            Error: If the file does already exist on the disk or does not exist on
                huggingface, an Error is raised.
        """
        if not os.path.isfile(self.model_checkpoint):
            path_to_model = hf_hub_download(
                repo_id="Hoebelt/macaw", filename=self.model_checkpoint
            )
            shutil.move(path_to_model, os.getcwd())
