"""
    A python implementation of the SSIM and NSSIM based on Zhang et al., 2018. No-reference blurred image quality
    assessment by structural similarity index. Applied Sciences (Switzerland), 8(10), 1-17.

    Folder structure for input images:
    >Main folder
        >class_1
            > img1
            > img2
            .
            .
            > img_n
        >class_2
            > img1
            > img2
            .
            .
            > img_n
        .
        .
        >class_n

"""

from PIL import Image
import numpy as np
import cv2
from skimage.util.shape import view_as_blocks
import os
import csv



def reblur(inp_img):
    """
        Apply a Gaussian blur kernel with kernel deviation of 1.5 and kernel size (11,11)
    :param inp_img:
    :return:
    """

    img = np.array(inp_img)
    kernel_deviation = 1.5
    y_img = cv2.GaussianBlur(img, (11, 11), kernel_deviation)

    return y_img


def downsampling(inp_img):
    """
        Image downsampling. Both the reblurred and the original images are downsampled using a low-pass filter and
        their channels are separated to a horizontal 2-dimensional stack.
    :param inp_img:
    :return:
    """


    img = np.array(inp_img)
    f = max(1, np.rint(np.amin(img)/256))

    if f > 1:
        lpf = np.ones((f, f))
        f = (1/(f*f))*lpf
        img = cv2.filter2D(img, -1, kernel=f)
    out = np.hstack((img[:, :, 0], img[:, :, 1], img[:, :, 2]))

    return out


def blurriness(inp_img):
    """
        d = ∑ p(gi)w(gi)

        where d represents image blurriness, gi is gray value whose range varies from 0 to the dynamic range L
        (e.g., L = 255 for 8-bit image), p(gi) is the proportion of gi on the whole image, and w(gi) represents
        the weight of gi, where
        w(gi) = gi/µ, if gi < µ;
        or
        w(gi) = L−gi/L−µ, if gi >= µ

    :param inp_img: input image
    :return: blur_sum
    """


    img = np.array(inp_img)
    L = 255
    img_shape = img.shape
    img_size = img_shape[0]*img_shape[1]

    mean_gray = np.mean(img)

    img_hist, bin_edges = np.histogram(img, range=(0, 255))

    blur_sum = 0
    for (num, gi) in zip(img_hist, bin_edges):
        # divide the count of 'gi' values in the image by the total number of pixels to get a proportion:
        proportion_gi = num/img_size
        if gi < mean_gray:
            weight_gi = gi/mean_gray
        else:
            weight_gi = (L-gi)/(L-mean_gray)
        blur_sum += proportion_gi*weight_gi

    return blur_sum


def blurriness_index(x_img, y_img):
    """
        Final blurriness comparison index
    :param x_img:
    :param y_img:
    :return:
    """
    x_blur_score, y_blur_score = blurriness(x_img), blurriness(y_img)

    const_4 = 0.03

    blurriness_comparison = (2*x_blur_score*y_blur_score + const_4)/(x_blur_score**2 + y_blur_score**2 + const_4)

    return blurriness_comparison


def luminance(x_img, y_img, N_x, N_y):
    """
        Luminance comparison index
    :param x_img:
    :param y_img:
    :param N_x:
    :param N_y:
    :return:
    """

    sum_x = np.sum(x_img)  # total luminance of x_img
    sum_y = np.sum(y_img)  # total luminance of y_img
    mu_x = sum_x/N_x  # mean luminance of x_img
    mu_y = sum_y/N_y  # mean luminance of y_img
    k_1 = 0.01  # small constant
    L = 255  # dynamic range of grayscale images
    const_1 = (k_1*L)**2
    # comparison of two images' luminance values:
    luminance_comparison = (2*mu_x*mu_y + const_1)/(mu_x**2 + mu_y**2 + const_1)

    return luminance_comparison, mu_x, mu_y


def contrast(x_img, y_img, mu_x, mu_y, N_x, N_y):
    """
        Contrast comparison index
    :param x_img:
    :param y_img:
    :param mu_x:
    :param mu_y:
    :param N_x:
    :param N_y:
    :return:
    """
    N_x = N_x - 1
    N_y = N_y - 1

    x_sum = 0
    for row in x_img:
        for pix in row:
            distance_from_mean = (pix-mu_x)**2
            x_sum += distance_from_mean

    y_sum = 0
    for row in y_img:
        for pix in row:
            distance_from_mean = (pix - mu_y)**2
            y_sum += distance_from_mean
    sigma_x = (x_sum/N_x)**(1/2)  # standard deviation for x_img
    sigma_y = (y_sum/N_y)**(1/2)  # standard deviation for y_img

    k_2 = 0.03  # small constant
    L = 255  # dynamic range of grayscale images
    const_2 = (k_2 * L) ** 2

    # comparison of two images' contrast values:
    contrast_comparison = (2 * sigma_x * sigma_y + const_2) / (sigma_x ** 2 + sigma_y ** 2 + const_2)
    return contrast_comparison, sigma_x, sigma_y, const_2


def structure(x_img, y_img, mu_x, mu_y, sigma_x, sigma_y, N_x, N_y, const_2):
    """
        Structural comparison index
    :param x_img: original input image
    :param y_img: reblurred reference image
    :param mu_x: mean luminance of x_img
    :param mu_y: mean luminance of y_img
    :param sigma_x: standard deviation of luminance for x_img
    :param sigma_y: standard deviation of luminance for y_img
    :param N_x: number of pixels in x_img
    :param N_y: number of pixels in y_img
    :param const_2: constant value from the contrast comparison, used to calculate const_3
    :return: structural correlation of the two images
    """
    cov_sum = 0
    for (row_x, row_y) in zip(x_img, y_img):
        for (pix_x, pix_y) in zip(row_x, row_y):
            pix_variance = (pix_x-mu_x)*(pix_y-mu_y)
            cov_sum += pix_variance

    covariance_xy = cov_sum/(N_x-1)

    const_3 = const_2 / 2  # small constant

    structure_correlation = (covariance_xy+const_3)/(sigma_x*sigma_y+const_3)
    return structure_correlation


def ssim(x_img, y_img, alpha=1, beta=1, gamma=1):
    """
        Structural similarity index
    :param x_img: original image
    :param y_img: reference image
    :param alpha: exponent coefficient of luminance
    :param beta: exponent coefficient of contrast
    :param gamma: exponent coefficient of structural correlation
    :return:
    """
    try:
        x_img = Image.open(x_img)
        y_img = Image.open(y_img)

        x_img = x_img.convert("L")  # convert x_img to grayscale
        y_img = y_img.convert("L")  # convert y_img to grayscale

        x_img = np.array(x_img)
        y_img = np.array(y_img)
    except AttributeError:
        #print("Images already open, you're probably applying NSSIM - passing through")
        pass
    x_img_shape = x_img.shape
    y_img_shape = y_img.shape

    N_x = x_img_shape[0] * x_img_shape[1]  # compute number of pixels in x_img
    N_y = y_img_shape[0] * y_img_shape[1]  # compute number of pixels in y_img

    try:
        assert N_x == N_y

    except AssertionError:
        print("Images are not the same size, rescale first")

    # luminance
    lum, mu_x, mu_y = luminance(x_img, y_img, N_x, N_y)
    lum = pow(lum, alpha)

    # contrast
    cont, sigma_x, sigma_y, const_2 = contrast(x_img, y_img, mu_x, mu_y, N_x, N_y)
    cont = pow(cont, beta)

    # structural correlation
    strut = structure(x_img, y_img, mu_x, mu_y, sigma_x, sigma_y, N_x, N_y, const_2)
    strut = pow(strut, gamma)

    return lum, cont, strut


def new_ssim(x_img, y_img, delta=1):
    """
        The extended SSIM with blurriness index
    :param x_img: original image
    :param y_img: reblurred reference image
    :param delta: exponent coefficient of blurriness
    :return: new_ssim score
    """
    lum, cont, strut = ssim(x_img, y_img)
    blur = blurriness_index(x_img, y_img)
    blur = pow(blur, delta)

    fin_score = lum*cont*strut*blur

    return fin_score


def mssim(img, patch_shape=(16, 16)):
    """
        To better capture the blurriness in local areas of an image, we partition an image into P × P patches of the
        same size and compute the mean SSIM

        MSSIM(x, y) = 1/M ∑ SSIM (xi, yi)

        where M = P × P, and xi and yi are the i-th patches in x and y respectively.

    :param img: original image
    :param patches:
    :return:
    """

    # first, apply reblur
    img_y = reblur(img)
    # second, downsample the reblurred and original images
    img_x = downsampling(img)
    img_y = downsampling(img_y)
    # third, create patches of the downsampled original and the reblurred image
    x_patches = view_as_blocks(np.array(img_x), block_shape=patch_shape)
    y_patches = view_as_blocks(np.array(img_y), block_shape=patch_shape)
    # calculate the sum of the similarity scores for all patches
    new_ssim_sum = 0
    for enum, (x,y) in enumerate(zip(x_patches, y_patches)):
        print("enum", enum)
        img_x = x[enum]
        img_y = y[enum]

        new_ssim_sum += new_ssim(img_x, img_y)

    mssim_measure = new_ssim_sum/(enum+1)
    return 1-mssim_measure

# change these:
folder = r""  # this is the main folder referenced at the top of this document in the first comment block
output_folder = r""  # the folder where you want the img_qual.csv file to be placed



lst_of_scores = []
for i in os.listdir(folder):
    indidividual = os.path.join(folder, i)
    for image in os.listdir(indidividual):
        img_name = os.path.join(indidividual, image)
        img = Image.open(img_name)
        img = img.convert("RGB")
        score = mssim(img)
        score_dict = {}
        score_dict['filename'] = str(img_name)
        score_dict['class'] = str(i)
        score_dict['mssim_score'] = str(score)
        lst_of_scores.append(score_dict)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(os.path.join(output_folder, 'img_qual.csv'), 'a+') as csvfile:
    fieldnames = ['filename', 'class', 'mssim_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|', lineterminator='\n')

    writer.writeheader()
    for row in lst_of_scores:
        writer.writerow(row)

