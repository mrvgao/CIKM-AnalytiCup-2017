from __parse_data import save_data_to_pickle, read_data
from __normalize_data import normalize_batch

save_data_to_pickle(read_data(test=False), test=False)
normalize_batch()
print('data load done:')