import logging

# Setup Logging #
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)
logging.info("Logging initialized")

from DataHandler import DataHandler
from LSTMClassificationModel import LSTMClassificationModel
from Settings import Settings

# Get Data From Dataset "
data_handler = DataHandler(Settings.classes,Settings.data_path)
data = data_handler.load_data(Settings.window_size)

# Create and Train Model #
model = LSTMClassificationModel(Settings.model_name,Settings.model_path)
model.create_model((Settings.window_size,Settings.dimensions),data,Settings.classes,Settings.epochs,Settings.batch_size)
