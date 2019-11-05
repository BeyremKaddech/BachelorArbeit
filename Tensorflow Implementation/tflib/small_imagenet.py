import numpy as np
from PIL import Image
import time

def make_generator(path, n_files, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = list(range(n_files))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = Image.open("{}/{}.jpg".format(path, i))
            image = np.transpose(image, (2,0,1))
            images[n % batch_size] = image
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='/var/tmp/ga94pak/Code2'):
    return make_generator(data_dir+'/faces+64', 13500, batch_size) #change

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print ("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()