import logging
import numpy as np
import random
import torch


def creat_log(loggerName: str, logPath: str):

    # create logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(level=logging.DEBUG)

    # create file handler
    file_handler = logging.FileHandler(logPath)
    file_handler.setLevel(level=logging.INFO)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # create formatter
    fmt = "[%(asctime)s] %(filename)s - %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def seed_everything(seed):
	random.seed(seed)
	# os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)         # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True



if __name__ == '__main__':

    logger = creat_log("test", "./log_test.log")

    logger.debug('debug')
    logger.info('info')
    logger.warning('waring')
    logger.error('error')
    logger.critical('critical')
