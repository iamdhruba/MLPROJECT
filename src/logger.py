import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%M_%d_%Y_%H_%M_%S')}.log"
log_dir = os.path.join(os.getcwd(),"logs")
os.makedirs(log_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# if __name__ == "__main__":
#     logging.info('Logging has Started')