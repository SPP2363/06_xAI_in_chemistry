import os
import pathlib

from pytorch_lightning import Callback


PATH: str = pathlib.Path(__file__).parent.absolute()
VERSION_PATH: str = os.path.join(PATH, 'VERSION')


def get_version() -> str:
    """
    Returns the version of the package as specified in the VERSION file.
    """
    with open(VERSION_PATH, 'r') as file:
        version_string: str = file.read().strip()
        version_string = version_string.replace('\n', '')
        
        return version_string


class PrintLossCallback(Callback):
    """A PyTorch Lightning callback that prints training loss to stdout at regular intervals.

    Args:
        n_epochs: Print loss every N epochs. Default is 5.
    """

    def __init__(self, n_epochs: int = 5):
        super().__init__()
        self.n_epochs = n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        current_epoch = trainer.current_epoch

        # Only print at specified intervals (and always at epoch 0)
        if current_epoch % self.n_epochs == 0 or current_epoch == 0:
            # Get the training loss from the logged metrics
            train_loss = trainer.callback_metrics.get('train_loss')
            val_loss = trainer.callback_metrics.get('val_loss')

            # Build the output message
            msg_parts = [f"Epoch {current_epoch}"]

            if train_loss is not None:
                msg_parts.append(f"train_loss: {train_loss:.4f}")

            if val_loss is not None:
                msg_parts.append(f"val_loss: {val_loss:.4f}")

            if len(msg_parts) > 1:  # Only print if we have loss values
                print(" | ".join(msg_parts))

    
        