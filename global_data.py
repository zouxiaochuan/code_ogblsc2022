import common_utils

DATA = None

def init(path):
    global DATA
    DATA = common_utils.load_obj(path)
    pass