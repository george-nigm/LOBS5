import time
import numpy as np
from glob import glob

book_files = sorted(glob("/homes/80/kang/LOBS5/GOOG2018/" + '*orderbook*.npy'))
print(len(book_files))
for file in book_files:
    print(file)
    x=np.load(file)
    assert(x.shape[1]==41), str(x.shape[1])+str(file)

