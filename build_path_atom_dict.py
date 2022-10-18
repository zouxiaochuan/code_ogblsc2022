from config import config 
import datasets
from tqdm import tqdm
import multiprocessing as mp
import multiprocessing.dummy as mpd
import common_utils
import os

ngram = 3

def process_func(param):
    i, dataset = param
    g = dataset[i][0]
    path = g['shortest_path']
    uniq_paths = set()
    for row in path:
        for path in row:
            path_atom = [g['atom_feat_cate'][a, 0] for a in path]
            
            if len(path_atom) <= 3:
                uniq_paths.add(tuple(path_atom))
            else:
                for i in range(len(path_atom) - ngram):
                    uniq_paths.add(tuple(path_atom[i:i+ngram]))
                    pass
                pass
            pass
        pass
    
    # q.put(i)
    return uniq_paths



if __name__ == '__main__':
    dataset_train = datasets.SimplePCQM4MDataset(path=config['middle_data_path'], split_name='train', rotate=True)

    bar = tqdm(range(len(dataset_train)))
    pool = mp.Pool()
    upaths = set()
    # m = mp.Manager()
    # q = m.Queue()
    # mpd.Process(target=thread_func, args=(q, bar)).start()
    for r in pool.imap(process_func, ((i, dataset_train) for i in bar), chunksize=5012):
        upaths.update(r)
        bar.set_postfix(length=len(upaths))
        pass
    pool.close()

    pathmap = {path: i for i, path in enumerate(upaths)}
    common_utils.save_obj(pathmap, os.path.join(config['middle_data_path'], 'path_atom_map.pkl'))
    pass