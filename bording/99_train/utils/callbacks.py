from logging import getLogger

from keras.callbacks import Callback
from abeja.train.client import Client
from abeja.train.statistics import Statistics as ABEJAStatistics

logger = getLogger('callback')


class Statistics(Callback):
    """cf. https://keras.io/callbacks/"""
    def __init__(self):
        super(Statistics, self).__init__()
        self.client = Client()

    def on_epoch_end(self, epoch, logs=None):
        epochs = self.params['epochs']
        statistics = ABEJAStatistics(num_epochs=epochs, epoch=epoch + 1)
        statistics.add_stage(ABEJAStatistics.STAGE_TRAIN, logs['acc'], logs['loss'])
        statistics.add_stage(ABEJAStatistics.STAGE_VALIDATION, logs['val_acc'], logs['val_loss'])
        try:
            self.client.update_statistics(statistics)
        except Exception:
            logger.warning('failed to update statistics.')
