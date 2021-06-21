import sys
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def split_image_into_tiles(image, tile_size = 16):
    tiles = []
    rows, cols = image.shape[0] // tile_size, image.shape[1] // tile_size

    for i in range(rows):
        for j in range(cols):
            tiles.append(image[i * tile_size: (i+1) * tile_size, j * tile_size: (j+1) * tile_size])

    return np.array(tiles)


def compute_lbp_value(lbp_window):
    lbp_value = 0
    pixel_value = lbp_window[1][1]

    index_table = [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1)]

    for i, index in enumerate(index_table):
        row_index, col_index = index
        lbp_value = lbp_value | (int(lbp_window[row_index][col_index] >= pixel_value) << i)

    return lbp_value


def pad_tile(tile):

    # We create a zeros matrix and we insert the tile inside with an offset
    rows, cols = tile.shape
    padded_tile = np.zeros((rows + 2, cols + 2))
    padded_tile[1 : -1, 1 : -1] = tile

    return padded_tile


def compute_texton_histogram(tile):

    # Step 3 Compute the histogram of the frequency of each texton occurring.

    texton_histogram = np.zeros(256)
    size = tile.shape[0]
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            lbp_window = tile[i - 1: i + 2, j - 1: j + 2]
            texton_histogram[compute_lbp_value(lbp_window)] += 1

    return texton_histogram / texton_histogram.max()



def compute_image_patch_histogram(tiles):

    image_histo = []

    for i, tile in enumerate(tiles):
        texton_histogram = compute_texton_histogram(pad_tile(tile))
        image_histo.append(texton_histogram)
        print(f"{i} / {len(tiles)}")

    return np.array(image_histo)


def random_lut(n_values):
    '''Build a random LUT for `n_values` elements (sequential integers).'''
    samples = np.linspace(0, 1, n_values)  # take n_values values between 0 and 1 (evenly spaced)
    rng = np.random.default_rng(3)  # get a RNG with a specific seed
    samples = rng.permutation(samples)  # shuffle our values
    colors = cm.hsv(samples, alpha=None, bytes=True)  # get corresponding colors from the HSV color map
    return colors[...,:3]  # remove alpha channel and return


def lbp(image, tile_size = 16, n_cluster = 16):

    tiles = split_image_into_tiles(image, tile_size)
    histogram = compute_image_patch_histogram(tiles)


    np.save(f"histo{tile_size}.npy", histogram)

    #histogram = np.load(f"histo{tile_size}.npy")

    kmeans = KMeans(n_clusters=n_cluster, random_state=128).fit(histogram)
    nearest_centroid = kmeans.predict(histogram)


    neigh = KNeighborsClassifier(n_neighbors = 1, metric="euclidean")
    neigh.fit(histogram, nearest_centroid)

    neigh_prediction = neigh.predict(histogram).reshape(image.shape[0] // tile_size, image.shape[1] // tile_size)

    lut = random_lut(n_cluster)
    recolored_image = lut[neigh_prediction]

    return recolored_image


if __name__ == "__main__":

    image_path = sys.argv[1]
    image = io.imread(image_path, as_gray=True)

    image_color = lbp(image, 16, 16)
    cv2.imwrite("filename.png", image_color)

    plt.imshow(image_color)
    plt.show()
