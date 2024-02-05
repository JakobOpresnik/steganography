import sys
import numpy as np
import cv2
import zlib
import random
import matplotlib.pyplot as plt
from scipy.stats import entropy


# function for plotting a histogram of pixel intensity
def plot_pixel_intensity_histogram(img_path, title):
    # read grayscale image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # calculate histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    plt.figure(figsize=(8, 6))
    plt.bar(bins[:-1], histogram, width=1)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


# calculate PSNR (Peak Signal-to-Noise Ratio) for original & modified images
def calculate_PSNR(original_path, modified_path):
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    modified_img = cv2.imread(modified_path, cv2.IMREAD_GRAYSCALE)

    if original_img.dtype != modified_img.dtype:
        raise ValueError("both images must have the same data type")
    
    # calculate MSE (Mean Square Error)
    mse = np.mean((original_img - modified_img) ** 2)
    if original_img.max() > 1.0:
        MAX = 255.0
    else:
        MAX = 1.0
    
    psnr = 10 * np.log10((MAX ** 2) / mse)
    return psnr


# calculate Shannon's entropy
def calculate_shannon_entropy(original_path):
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    flattened_img = original_img.flatten()
    histogram, _ = np.histogram(flattened_img, bins=256, range=[0, 256])
    probabilities = histogram / float(np.sum(histogram))
    entropy_value = entropy(probabilities, base=2)
    return entropy_value


# calculate modified image blockiness
def calculate_blockiness(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    height, width = img.shape

    B = 0

    # blockiness along rows
    for i in range((height - 1) // 8):
        for j in range(width):
            B += abs(img[8*i, j] - img[8*i+1, j])

    # blockiness along columns
    for j in range((width - 1) // 8):
        for i in range(height):
            B += abs(img[i, 8*j] - img[i, 8*j+1])
    return B


# convert raw binary string first to hex, then to plain text
def binary_to_message(extracted_message_bin):
    print("extracted binary message:", extracted_message_bin)
    byte_array = bytes.fromhex(format(int(extracted_message_bin, 2), 'x'))
    return byte_array.decode('utf-8')


# convert plain text to raw binary
def binarize_message(msg):
    bin_msg = ''.join(format(ord(char), '08b') for char in msg)
    bin_size = len(bin_msg)

    print("bin message:", bin_msg)

    size_prefix = format(bin_size, '032b')
    bin_msg_with_prefix = size_prefix + bin_msg

    print("bin message with prefix:", bin_msg_with_prefix)
    return bin_msg_with_prefix


# array of 8x8 matrices --> zigzag 1D array
def zigzag_transform(matrices, N):
    result_array = []
    rows, cols = matrices.shape[1:]

    for x in range(len(matrices)):
        for i in range(rows + cols - 1):
            if i % 2 == 0:
                # even rows go up
                for row in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                    result_array.append(matrices[x][row, i - row])
            else:
                # odd rows go down
                for row in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    result_array.append(matrices[x][row, i - row])
        
        # thresholding (set last N elements to zero)
        for c in range(-N, 0):
            result_array[c] = 0

    return result_array


# image split into array 8x8 blocks for each color channel
def split_img_into_blocks(img, block_size):
    blocks = []
    height, width = img.shape
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img[i:i+block_size, j:j+block_size]

            blocks.append(block)
    
    blocks = np.array(blocks)
    return blocks


# finds least significant bits for each trio
def find_LSB(array):
    C = []
    for i in array:
        # get least significant bit
        lsb = i & 1
        C.append(lsb)
    return C


def negate_LSB(coefficient):
    return coefficient ^ 1


def alg_F5(array, N, M, msg):
    # remove last N elements
    array = array[:-N]

    # set to track selected elements
    selected_elements = set()

    for _ in range(M):
        # set upper limit for random number generation
        upper_limit = 32 - 3
        if N > 32: upper_limit = 64 - 3 - N

        # select 3 non-overlapping elements within the interval [4, 32]
        selected_indices = random.sample(range(4, upper_limit), 3)

        # check and redo the selection if any element has already been selected (ensuring uniqueness)
        while any(i in selected_elements for i in selected_indices):
            selected_indices = random.sample(range(4, upper_limit), 3)

        # update the set of selected elements
        selected_elements.update(selected_indices)

        # append the selected elements to the result array
        AC_array = [array[i] for i in selected_indices]

        # get array of least significant bits for each trio
        C = find_LSB(AC_array)

        # get 2 random bits from hidden message
        selected_x_indices = random.sample(range(len(msg)), 2)
        x1, x2 = int(msg[selected_x_indices[0]]), int(msg[selected_x_indices[1]])

        # destructure C array of LSBs
        c1, c2, c3 = C

        if x1 != c1 ^ c2 and x2 == c2 ^ c3:
            AC_array[0] = negate_LSB(AC_array[0])
        if x1 == c1 ^ c2 and x2 != c2 ^ c3:
            AC_array[2] = negate_LSB(AC_array[2])
        if x1 != c1 ^ c2 and x2 != c2 ^ c3:
            AC_array[1] = negate_LSB(AC_array[1])

        i1, i2, i3 = selected_indices

        # update zig-zagged array
        array[i1] = AC_array[0]
        array[i2] = AC_array[1]
        array[i3] = AC_array[2]

    # add zeros that were removed at the start of this function
    missing_zeros = np.zeros(N, dtype=array.dtype)
    full_array = np.append(array, missing_zeros).astype(np.int32)
    return full_array
        

# zigzag 1D array --> array of 8x8 matrices
def inverse_zigzag_transform(zigzag_array, shape):
    result = np.zeros(shape)
    indices = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    for matrix in range(shape[0]):
        for i, index in enumerate(indices):
            result[matrix, index[0], index[1]] = zigzag_array[matrix * len(indices) + i]

    return result


# array of 8x8 blocks reconstructed into original padded image
def reconstruct_image_from_blocks(blocks, height, width):
    block_size = 8
    reconstructed_img = np.zeros((height, width), dtype=np.float32)

    for i, block in enumerate(blocks):
        num_blocks = width // block_size
        row = i // num_blocks * block_size
        col = (i % num_blocks) * block_size
        reconstructed_img[row : row + block_size, col : col + block_size] = block

    return reconstructed_img


# extract hidden message from modified image (inverse F5 algorithm)
def extract_message(array, msg_size, selected_elements, N):

    # set upper limit for random number generation
    upper_limit = 32 - 3
    if N > 32: upper_limit = 64 - 3 - N

    # select 3 non-overlapping elements within the interval [4, 32]
    selected_indices = random.sample(range(4, upper_limit), 3)

    # check and redo the selection if any element has already been selected (ensuring uniqueness)
    while any(i in selected_elements for i in selected_indices):
        selected_indices = random.sample(range(4, upper_limit), 3)

    selected_elements.update(selected_indices)

    AC_array = [array[i] for i in selected_indices]

    C = find_LSB(AC_array)
    c1, c2, c3 = C

    x1 = c1 ^ c2
    x2 = c2 ^ c3

    selected_x_indices = random.sample(range(msg_size), 2)
    i1, i2 = selected_x_indices

    return i1, i2, x1, x2



def compress(path_in, H, N, M, bin_msg):
    original_img = cv2.imread(path_in)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    print(f"image shape: {original_img.shape}")
    print(f"image data type: {original_img.dtype}")

    height, width = original_img.shape[:2]
    print(f"dimensions: {width}x{height}")

    new_height = ((height+7) // 8) * 8
    new_width = ((width+7) // 8) * 8

    # add padding
    padded_img = np.zeros((new_height, new_width), dtype=np.uint8)
    padded_img[0:height, 0:width] = original_img

    print(f"dimensions after padding: {new_width}x{new_height}")

    # split padded image into arrays of 8x8 blocks
    blocks = split_img_into_blocks(padded_img, block_size=8)

    #print("1. block:\n", blocks[0])
    
    H_transposed = H.T

    # Haar transformation
    tmp = H_transposed @ blocks
    haar_blocks = tmp @ H

    #print("1. block after haar:\n", haar_blocks[0])

    # zig-zag transform
    haar_array = np.array(zigzag_transform(haar_blocks, N), dtype=np.float64)

    #print("1. block after zigzag:\n", haar_array[0:64])

    # round coefficients to integers
    haar_array = np.round(haar_array).astype(np.int32)

    #print("1. block after zigzag to int:\n", haar_array[0:64])

    # split entire data to list of zig-zagged arrays
    chunk_size = 64
    haar_arrays = np.array_split(haar_array, len(haar_array) // chunk_size)

    random.seed(N*M)
    combined_arrays = []
    msg_size = len(bin_msg)
    size = 0
    # hide part of the message into each zig-zagged array
    for i in range(len(haar_arrays)):
        if size <= msg_size:
            haar_arrays[i] = alg_F5(haar_arrays[i], N, M, bin_msg)
            size += 1
        combined_arrays.append(haar_arrays[i])
    
    print("1. block after F5:\n", combined_arrays[0])

    # combine all data
    combined_data = np.concatenate((combined_arrays)).astype(np.int32)

    # encode & compress using zlib
    data = zlib.compress(combined_data)

    # calculate compression rate
    original_size = sys.getsizeof(original_img)
    compressed_size = sys.getsizeof(data)
    compression_rate = original_size / compressed_size
    print(f"original image size: {original_size}")
    print(f"compressed size: {compressed_size}")
    print(f"compression rate: {compression_rate:.2f}")

    # save to file
    with open("compressed.bin", "wb") as compressed_file:
        compressed_file.write(new_width.to_bytes(2, byteorder='big'))
        compressed_file.write(new_height.to_bytes(2, byteorder='big'))
        compressed_file.write(data)
    compressed_file.close()


def decompress(path_out, H, N, M, msg_size):
    # read compressed file
    with open("compressed.bin", "rb") as bin_file:
        width_bytes = bin_file.read(2)
        height_bytes = bin_file.read(2)
        compressed_data = bin_file.read()
    bin_file.close()

    original_width = int.from_bytes(width_bytes, byteorder='big')
    original_height = int.from_bytes(height_bytes, byteorder='big')
    print(f"original dimensions: {original_width}x{original_height}")

    # decode & decompress using zlib
    decompressed_bytes = zlib.decompress(compressed_data)
    decompressed = np.frombuffer(decompressed_bytes, dtype=np.int32)

    #print("1. block after decompression:\n", decompressed[0:64])

    # split all data into multiple zig-zagged arrays
    chunk_size = 64
    haar_arrays = np.array_split(decompressed, len(decompressed) // chunk_size)


    # extract message from modified image
    extracted_binary_sequence = '_' * msg_size

    random.seed(N*M)
    size = 0
    for array in haar_arrays:
        if size <= msg_size:
            selected_elements = set()
            for _ in range(M):
                i1, i2, x1, x2 = extract_message(array, msg_size, selected_elements, N)
                extracted_binary_sequence = extracted_binary_sequence[:i1] + str(x1) + extracted_binary_sequence[i1+1:]
                extracted_binary_sequence = extracted_binary_sequence[:i2] + str(x2) + extracted_binary_sequence[i2+1:]
            size += 1
    
    # convert extracted binary message to plain text
    extracted_message_bin = extracted_binary_sequence[32:]
    extracted_message = binary_to_message(extracted_message_bin)

    print("extracted message:\n", extracted_message)

    num_blocks = (original_width * original_height) // (8 * 8)

    haar = np.concatenate(haar_arrays)

    # inverse zig-zag transformation
    haar_blocks = inverse_zigzag_transform(haar, (num_blocks, 8, 8))

    H_transposed = H.T

    # inverse Haar transformation
    tmp = H @ haar_blocks
    blocks = tmp @ H_transposed

    # reconstruct image from 8x8 blocks
    reconstructed_img = reconstruct_image_from_blocks(blocks, original_height, original_width)

    cv2.normalize(reconstructed_img, reconstructed_img, 0, 255, cv2.NORM_MINMAX)
    reconstructed_img = reconstructed_img.astype(np.uint8)
    #reconstructed_img = reconstructed_img[:, :-4]
    #reconstructed_img = reconstructed_img[:-4, :]


    # show reconstructed image
    cv2.imshow("decompressed image: ", reconstructed_img)
    cv2.imwrite(path_out, reconstructed_img)
    cv2.waitKey(0)