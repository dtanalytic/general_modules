import logging
import pickle


def setup_logger(name, formatter, log_file, level=logging.INFO):

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(formatter))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def save_struct(filename,struct_to_save):
    with open(filename , 'wb') as f:
        pickle.dump(struct_to_save,f)

def load_struct(filename):
    with open(filename , 'rb') as f:
        struct_to_load = pickle.load(f)
    return struct_to_load