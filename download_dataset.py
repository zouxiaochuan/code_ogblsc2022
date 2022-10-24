
from config import config
from ogb.lsc import PCQM4Mv2Dataset
import os

if __name__ == '__main__':
    # download pcqm4m-v2
    os.makedirs(config['ogb_data_path'], exist_ok=True)
    dataset = PCQM4Mv2Dataset(root = config['ogb_data_path'], only_smiles=True)

    # download sdf
    os.system('wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz -P {0}'.format(config['ogb_data_path']))
    os.system('tar -xf {0}/pcqm4m-v2-train.sdf.tar.gz -C {0}'.format(config['ogb_data_path']))
    pass