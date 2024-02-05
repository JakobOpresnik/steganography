import math
from utils import *


args = sys.argv[1:]

if len(args) == 5:
    # path to original image
    img_path = "images/" + args[0]

    # h - hide message OR e - extract message
    option = args[1]

    # path to message
    msg_path = args[2]

    # threshold
    N = int(args[3])

    # number of unique coefficient trios used in F5 steganography algorithm
    M = int(args[4])

    print(f"input arguments: {args}")

    with open(msg_path, "r") as msg_file:
        message = msg_file.read()
    msg_file.close()

    print(message)
    bin_message = binarize_message(message)
    msg_size = len(bin_message)
    print(f"message size: {msg_size}")

    # Haar transformation
    root = math.sqrt(8/64)
    small_root = math.sqrt(2/4)
    fraction = 1/2
    # Haar's matrix
    H = np.array([
        [root, root, fraction, 0, small_root, 0, 0, 0],
        [root, root, fraction, 0, -small_root, 0, 0, 0],
        [root, root, -fraction, 0, 0, small_root, 0, 0],
        [root, root, -fraction, 0, 0, -small_root, 0, 0],
        [root, -root, 0, fraction, 0, 0, small_root, 0],
        [root, -root, 0, fraction, 0, 0, -small_root, 0],
        [root, -root, 0, -fraction, 0, 0, 0, small_root],
        [root, -root, 0, -fraction, 0, 0, 0, -small_root]
    ])

    if option == 'h':
        compress(img_path, H, N, M, bin_message)

        plot_pixel_intensity_histogram(img_path, "Original Image")

        sys.exit(0)
    if option == 'e':
        decompress(img_path, H, N, M, msg_size)

        plot_pixel_intensity_histogram(img_path, "Modified Image")
        
        # calculate PSNR
        psnr = calculate_PSNR("images/puppy.bmp", "images/decomp_puppy.bmp")
        print(f"PSNR metric: {psnr}")

        # calculate Shannon's entropy
        entropy_original = calculate_shannon_entropy("images/puppy.bmp")
        entropy_modified = calculate_shannon_entropy("images/decomp_puppy.bmp")
        entropy_avg = (entropy_original + entropy_modified) / 2
        print(f"Shannon's entropy: {entropy_avg}")

        # calculate blockiness
        blockiness_original = calculate_blockiness("images/puppy.bmp")
        blockiness_modified = calculate_blockiness("images/decomp_puppy.bmp")
        blockiness_avg = (blockiness_original + blockiness_modified) / 2

        print(f"blockiness: {blockiness_avg}")

        sys.exit(0)
    
    print("ERROR: incorrect 2. argument - use 'h' for hiding message & compression or 'e' for message extraction & decompression")

else: print("ERROR: 5 arguments needed")




# python main.py puppy.bmp h message.txt 20 3
# python main.py decomp_puppy.bmp e message.txt 20 3