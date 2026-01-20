import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%M_%D_%Y_%H_%M_%S')}.log"
log_paths = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_paths, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_paths, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d - %(levelname)s - %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info('Logging has Started')