import logging

import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config 


def build_logger():
    log_file = config.LOG_FILE 

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(filename=log_file, mode="a", encoding="utf-8")
    formatting = logging.Formatter("%(asctime)s -> %(levelname)s - %(message)s")

    file_handler.setFormatter(formatting)

    return logger, file_handler

