#import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.utils import shuffle
import os, os.path , sys
from keras.layers import Input, Dense
from keras.models import Model
import colorsys
import scipy
import logging
from scipy.misc import imread
from skimage import color
from skimage.filters import roberts, sobel, scharr, prewitt
import math
# Cuts the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(precision=8, suppress=True)

images_path = "" #where images wait to be encoded
decoded_images_path = "" # decoded images


# gets image path from the folder
def get_image_path(path):
    valid_images = (".jpg", ".png", ".tiff", ".tif")  # supported image types
    for f in os.listdir(images_path):
        if f.lower().endswith(valid_images):
            return path + f


# Load image
or_image = Image.open(get_image_path(images_path))

# Palette color count after color quantization
n_colors = 32

# __Normalization__
# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 to be in the range [0-1] is important so that machine learning algorithms and plt.imshow works well
or_image = np.array(or_image, dtype=np.float64) / 255
# decides image is grayscale or rgb format
is_org_img_rgb = len(or_image.shape) > 2
# print(or_image.shape)
if is_org_img_rgb:

    # Gets image shape data for later usage
    w, h, d = original_shape = tuple(or_image.shape)
    # Handle delta channel(Transparency)
    if (or_image.shape[2] == 4):
        assert d == 4
    else:
        assert d == 3

    for i in range(10):
        if w%8 != 0:
            w = w - 1
        if h%8 != 0:
            h = h - 1

    # print(w, h)
    # print(or_image.shape)
    or_image = or_image[:w, :h]
    # print(or_image.shape)

    image_array = np.reshape(or_image, (w * h, d))

else:
    w, h = original_shape = tuple(or_image.shape)
    d = 1
    image_array = np.reshape(or_image, (w * h, d))

    # width_pixel = math.ceil(float(w)/float(8)) * 8
    # height_pixel = math.ceil(float(h)/float(8)) * 8
    #
    # print(image_array.shape)
    # image_array = image_array[:width_pixel,:height_pixel]
    # print(image_array.shape)




# Fitting model on a small sub-sample of the data
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

# Get labels for all points
# Predicting color indices on the full image (k-means)
labels = kmeans.predict(image_array)


# Recreates the quantized graysale image from the code book & labels
def recreate_image_gray(codebook, labels, w, h):
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Recreate the quantized colorful image from the code book & labels
def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# RGB to YCbCr color space
def _ycc(r, g, b):  # in (0,255) range
    y = .299 * r + .587 * g + .114 * b
    cb = 128 - .168736 * r - .331364 * g + .5 * b
    cr = 128 + .5 * r - .418688 * g - .081312 * b
    return y, cb, cr


# YCbCr to RGB color space
def _rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - .34414 * (cb - 128) - .71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return r, g, b


# Creates and stores quantized image
quantized_image = recreate_image(kmeans.cluster_centers_, labels, w, h) if is_org_img_rgb else recreate_image_gray(
    kmeans.cluster_centers_, labels, w, h)

# Maps image color values to luminance and if it exists chrominance values
if is_org_img_rgb:
    y, cb, cr = _ycc(quantized_image[:, :, 0], quantized_image[:, :, 1], quantized_image[:, :, 2])
else:
    y = or_image


# Tiles image into 8 by 8 pixel blocks
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


# Recreates image from 8 by 8 tiled form
def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))



#-------------------------------------------Look HERE------------------



# print(y.shape)
# y = y[:-1,:]
#
# print(y.shape)

# Tiles image to 8*8 blocks and stores
img_blocks = blockshaped(y, 8, 8)
# Stores tile count for later usage
img_blocks_number = len(img_blocks[:][:][:])
# Normalizes blocks from (8,8) 2D format to (64,1) 1D format
img_blocks = img_blocks.reshape(img_blocks_number, 64)

# ---------------------------------------------------------------Autoencoder Starts------------------------------------------






# Number of neurons in the hidden layer of the network (Its number decides the compression ratio)
encoding_dim = 16

# This is our input placeholder
input_img = Input(shape=(64,))
# "encoded" is the encoded representation of the input (Output of the hidden layer)
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input (final output of the neural network)
decoded = Dense(64, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# Creates a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# Retrieves the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# This model maps an encoder output to final output
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Training neural network
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(img_blocks, img_blocks, epochs=1000, batch_size=1024)
encoded_imgs = encoder.predict(img_blocks)
decoded_imgs = decoder.predict(encoded_imgs)

# Gets height and width of image for later usage
h, w = y.shape
# print(y.shape)

# Reshaping output image
decoded_imgs = decoded_imgs.reshape(img_blocks_number, 8, 8)
decoded = unblockshaped(np.array(decoded_imgs), h, w)

# Maps ycbcr to rgb color space
if is_org_img_rgb:
    r, g, b = _rgb(decoded, cb, cr)
    rgbArray = np.zeros((h, w, 3), 'uint8')
    rgbArray[..., 0] = r * 256
    rgbArray[..., 1] = g * 256
    rgbArray[..., 2] = b * 256
    img = Image.fromarray(rgbArray)
else:
    img = decoded * 256

    # Show images
    fig = plt.figure(figsize=(16, 13))
    fig.add_subplot(2, 2, 1).set_title('Original')
    imgplot = plt.imshow(or_image, cmap="gray")
    fig.add_subplot(2, 2, 2).set_title('Quantized')
    imgplot = plt.imshow(quantized_image, cmap="gray")
    fig.add_subplot(2, 2, 3).set_title('Autoencoder')
    imgplot = plt.imshow(img, cmap="gray")

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img.save(decoded_images_path + '/decoded.png')
    plt.show()
    sys.exit(0)

filter_image = img


# Creates an Image from decoded image which has luminance value and some important pixel colors
def getRepresentativeColorImage(dec_image):
    pic_o_rgb = np.array(dec_image)
    pic_o_rgb_gray = color.rgb2gray(pic_o_rgb)



    # Creates distorted image by the filters
    image = pic_o_rgb_gray
    edge_roberts = roberts(image)
    edge_sobel = sobel(image)

    filter_image = edge_roberts
    pic_m_rgb = edge_roberts

    pic_o_rgb_gray = pic_o_rgb_gray * 256

    w, h, c = pic_o_rgb.shape

    width, height, channel = pic_o_rgb.shape
    white_img = pic_o_rgb_white = np.full((width, height, channel), 255, dtype=int)

    # Gets color values if distorted grayscale image has black values corresponding to the colorful decoded image
    for row in range(pic_m_rgb.shape[0]):
        for col in range(pic_m_rgb.shape[1]):
            pic_o_rgb[row, col] = pic_o_rgb[row, col] if pic_m_rgb[row, col] == 0 else pic_o_rgb_gray[row, col]
            white_img[row, col] = pic_o_rgb[row, col] if pic_m_rgb[row, col] == 0 else pic_o_rgb_white[row, col]

    pic_o_rgb = Image.fromarray(pic_o_rgb)
    white_img = Image.fromarray(np.uint8(white_img * 255.999))

    pic_o_rgb.save(r"/home/utkumukan/Masaüstü/images/recreated.png")
    white_img.save(r"/home/utkumukan/Masaüstü/images/whitehints.png")

    # Returns regenerated image luminance value and important color pixel values
    return np.array(pic_o_rgb)


# ---------------------------------------------------------------Colorization Starts------------------------------------------




# the window class, find the neighbor pixels around the center.
class WindowNeighbor:
    def __init__(self, width, center, pic):
        # center is a list of [row, col, Y_intensity]
        self.center = [center[0], center[1], pic[center][0]]
        self.width = width
        self.neighbors = None
        self.find_neighbors(pic)
        self.mean = None
        self.var = None

    def find_neighbors(self, pic):
        self.neighbors = []
        ix_r_min = max(0, self.center[0] - self.width)
        ix_r_max = min(pic.shape[0], self.center[0] + self.width + 1)
        ix_c_min = max(0, self.center[1] - self.width)
        ix_c_max = min(pic.shape[1], self.center[1] + self.width + 1)
        for r in range(ix_r_min, ix_r_max):
            for c in range(ix_c_min, ix_c_max):
                if r == self.center[0] and c == self.center[1]:
                    continue
                self.neighbors.append([r, c, pic[r, c, 0]])

    def __str__(self):
        return 'windows c=(%d, %d, %f) size: %d' % (self.center[0], self.center[1], self.center[2], len(self.neighbors))


# affinity functions, calculate weights of pixels in a window by their intensity.
def affinity_a(w):
    nbs = np.array(w.neighbors)
    sY = nbs[:, 2]
    cY = w.center[2]
    diff = sY - cY
    sig = np.var(np.append(sY, cY))
    if sig < 1e-6:
        sig = 1e-6
    wrs = np.exp(- np.power(diff, 2) / (sig * 2.0))
    wrs = - wrs / np.sum(wrs)
    nbs[:, 2] = wrs
    return nbs


# translate (row,col) to/from sequential number
def to_seq(r, c, rows):
    return c * rows + r


def fr_seq(seq, rows):
    r = seq % rows
    c = int((seq - r) / rows)
    return (r, c)


# combine 3 channels of YUV to a RGB photo: n x n x 3 array
def yuv_channels_to_rgb(cY, cU, cV):
    ansRGB = [colorsys.yiq_to_rgb(cY[i], cU[i], cV[i]) for i in range(len(ansY))]
    ansRGB = np.array(ansRGB)
    pic_ansRGB = np.zeros(pic_yuv.shape)
    pic_ansRGB[:, :, 0] = ansRGB[:, 0].reshape(pic_rows, pic_cols, order='F')
    pic_ansRGB[:, :, 1] = ansRGB[:, 1].reshape(pic_rows, pic_cols, order='F')
    pic_ansRGB[:, :, 2] = ansRGB[:, 2].reshape(pic_rows, pic_cols, order='F')
    return pic_ansRGB


# rgb to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def init_logger():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logger = logging.getLogger()
    return logger


# window width
wd_width = 1

# Gets grayscale version of regenerated hint image for optimization function
pic_o_rgb = getRepresentativeColorImage(np.array(img))
pic_o = pic_o_rgb.astype(float) / 255

# Gets regenerated hint image for optimization function
pic_m_rgb = getRepresentativeColorImage(img)
pic_m = pic_m_rgb.astype(float) / 255

# prepare matrix A
(pic_rows, pic_cols, _) = pic_o.shape
pic_size = pic_rows * pic_cols

channel_Y, _, _ = colorsys.rgb_to_yiq(pic_o[:, :, 0], pic_o[:, :, 1], pic_o[:, :, 2])
_, channel_U, channel_V = colorsys.rgb_to_yiq(pic_m[:, :, 0], pic_m[:, :, 1], pic_m[:, :, 2])

map_colored = (abs(channel_U) + abs(channel_V)) > 0.0001

pic_yuv = np.dstack((channel_Y, channel_U, channel_V))
weightData = []
num_pixel_bw = 0

# build the weight matrix for each window.
for c in range(pic_cols):
    for r in range(pic_rows):
        res = []
        w = WindowNeighbor(wd_width, (r, c), pic_yuv)
        if not map_colored[r, c]:
            weights = affinity_a(w)
            for e in weights:
                weightData.append([w.center, (e[0], e[1]), e[2]])
        weightData.append([w.center, (w.center[0], w.center[1]), 1.])

sp_idx_rc_data = [[to_seq(e[0][0], e[0][1], pic_rows), to_seq(e[1][0], e[1][1], pic_rows), e[2]] for e in weightData]
sp_idx_rc = np.array(sp_idx_rc_data, dtype=np.integer)[:, 0:2]
sp_data = np.array(sp_idx_rc_data, dtype=np.float64)[:, 2]

matA = scipy.sparse.csr_matrix((sp_data, (sp_idx_rc[:, 0], sp_idx_rc[:, 1])), shape=(pic_size, pic_size))

# prepare vector b
b_u = np.zeros(pic_size)
b_v = np.zeros(pic_size)
idx_colored = np.nonzero(map_colored.reshape(pic_size, order='F'))
pic_u_flat = pic_yuv[:, :, 1].reshape(pic_size, order='F')
b_u[idx_colored] = pic_u_flat[idx_colored]

pic_v_flat = pic_yuv[:, :, 2].reshape(pic_size, order='F')
b_v[idx_colored] = pic_v_flat[idx_colored]

# optimize the problem
ansY = pic_yuv[:, :, 0].reshape(pic_size, order='F')
ansU = scipy.sparse.linalg.spsolve(matA, b_u)
ansV = scipy.sparse.linalg.spsolve(matA, b_v)
pic_ans = yuv_channels_to_rgb(ansY, ansU, ansV)

# Show images
fig = plt.figure(figsize=(20, 15))
fig.add_subplot(3, 3, 1).set_title('Original')
imgplot = plt.imshow(or_image)
fig.add_subplot(3, 3, 2).set_title('Quantized')
imgplot = plt.imshow(quantized_image)
fig.add_subplot(3, 3, 3).set_title('Autoencoder')
imgplot = plt.imshow(img)
fig.add_subplot(3, 3, 4).set_title('Filter')
imgplot = plt.imshow(filter_image)
fig.add_subplot(3, 3, 5).set_title('Representative Colors White')
imgplot = plt.imshow(Image.open(r"/home/utkumukan/Masaüstü/images/whitehints.png"))
fig.add_subplot(3, 3, 6).set_title('Encoded Image')
imgplot = plt.imshow(Image.open(r"/home/utkumukan/Masaüstü/images/recreated.png"))
fig.add_subplot(3, 3, 7).set_title('Colorized')
imgplot = plt.imshow(pic_ans)

img.save(decoded_images_path + '/decoded.png')
plt.show()