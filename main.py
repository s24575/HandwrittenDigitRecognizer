from mnist import MNIST
import os
import urllib.request
import gzip
from shutil import copyfileobj
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as pt


def download_unpack_and_save_file(file_url, file_name_gz, file_name):
    urllib.request.urlretrieve(file_url, file_name_gz)
    with gzip.open(file_name_gz, 'rb') as f_in:
        with open(file_name, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    os.remove(file_name_gz)


def download_data(directory):
    if not os.path.isdir(directory):
        print("Downloading data...")
        Path(directory).mkdir()
        download_unpack_and_save_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                                      directory + "train-images-idx3-ubyte.gz",
                                      directory + "train-images-idx3-ubyte")
        download_unpack_and_save_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                                      directory + "train-labels-idx1-ubyte.gz",
                                      directory + "train-labels-idx1-ubyte")
        download_unpack_and_save_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                                      directory + "t10k-images-idx3-ubyte.gz",
                                      directory + "t10k-images-idx3-ubyte")
        download_unpack_and_save_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                                      directory + "t10k-labels-idx1-ubyte.gz",
                                      directory + "t10k-labels-idx1-ubyte")


def main():
    download_data("./dataset/")
    print("Training...")
    mndata = MNIST('dataset')
    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(predictions, y_test))
    print(x_test)
    for i in range(0, 10):
        print(model.predict([x_test[i]]))
        d = np.array(x_test[i])
        d.shape = (28, 28)
        pt.imshow(255 - d, cmap='gray')
        pt.show()


if __name__ == '__main__':
    main()
