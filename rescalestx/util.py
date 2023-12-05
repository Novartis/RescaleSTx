import scanpy as sc
import pandas as pd
import os
import pdb
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import anndata
import squidpy as sq
import cv2 as cv
import matplotlib.pyplot as plt
import math


def plot_spot_radii_stats(mean_spot_radii_px):
    pass

def read_visium_object(metadata):
    # metadata is a pandas row
    folder_name = os.path.dirname(metadata['samples'])
    count_file = metadata['samples']
    adata = sq.read.visium(path = folder_name,counts_file=count_file, library_id = metadata['sample_names'],
                          source_image_path = metadata['imgs'])
    fields = metadata.index.tolist()
    fields.remove('samples')
    for field in fields:
        adata.obs[field] = [metadata[field]]*adata.n_obs
        adata.obs[field] = adata.obs[field].astype('category')

    adata.var['symbol'] = adata.var_names
    adata.var_names = adata.var['gene_ids']
    adata.var['symbol'] = adata.var['symbol'].where(~adata.var['symbol'].duplicated(), adata.var['symbol'].index + '_1')
    #adata.obs_names=["{0}-{1}".format(x, row['sample_names']) for x in adata.obs_names]
    adata.raw = adata
    return(adata)



def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def draw_moments(image, contours, contour_thickness=2, circle_thickness=-1, fontScale=0.5, text_thickness=2):
    img_contour = image.copy()
    moment_pairs = []
    for idx,i in enumerate(contours):
        M = cv.moments(i)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            moment_pairs.append((cx,cy))
            moment_pair_index = len(moment_pairs)-1
            img_contour = cv.drawContours(image=img_contour, contours=[i], contourIdx=-1, color=(0, 255, 0), thickness=contour_thickness)
            img_contour = cv.circle(img=img_contour, center=(cx, cy), radius=7, color=(0, 0, 255), thickness=circle_thickness)
            img_contour = cv.putText(img=img_contour, text=f"center{moment_pair_index}", org=(cx - 20, cy - 20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(0, 0, 0), thickness=text_thickness)
        print(f"x: {cx} y: {cy}")
    return(img_contour, moment_pairs)

def compare_moments(moment_pairs1, moment_pairs2, match_inds):
    # match_inds are indexes of the moments for each respective pair
    x_errors = []
    y_errors = []
    for match in match_inds:
        x_error = moment_pairs1[match[0]][0] - moment_pairs2[match[1]][0]
        y_error = moment_pairs1[match[0]][1] - moment_pairs2[match[1]][1]
        x_errors.append(x_error)
        y_errors.append(y_error)
    sse = np.sum(np.power(x_errors,2)) + np.sum(np.power(y_errors,2))
    return((x_errors, y_errors, sse))
def pipeline1(image,gblur_sigma=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (0, 0), sigmaX=gblur_sigma, sigmaY=gblur_sigma, borderType=cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(blur, 200, 255,cv.THRESH_BINARY_INV)
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return(thresh, contours)

def pipeline2(image,median_blur_ksize=101,contour_thickness=30, threshold_blocksize=11):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, median_blur_ksize)
    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,threshold_blocksize,2)
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(thresh.shape[:2],dtype='uint8')

    cv.drawContours(blank, contours, -1,(255, 0, 0), contour_thickness)

    fig = plt.figure(figsize=(28,15))
    ax1 = fig.add_subplot(1,4,1)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)

    ax1.imshow(gray)
    ax2.imshow(blur)
    ax3.imshow(thresh)
    ax4.imshow(blank)
    return(contours)

def pipeline3(image,gblur_sigma=101,contour_thickness=30, threshold_blocksize=11):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (0, 0), sigmaX=gblur_sigma, sigmaY=gblur_sigma, borderType=cv.BORDER_DEFAULT)
    thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,threshold_blocksize,2)
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(thresh.shape[:2],dtype='uint8')

    cv.drawContours(blank, contours, -1,(255, 0, 0), contour_thickness)

    fig = plt.figure(figsize=(28,15))
    ax1 = fig.add_subplot(1,4,1)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)

    ax1.imshow(gray)
    ax2.imshow(blur)
    ax3.imshow(thresh)
    ax4.imshow(blank)
    return(contours)

def get_spot_info(hr_image,template, median_blur_ksize=31, rectangle_thickness = 30, radii_threshold = 5, blur_filter = 'median', threshold_blocksize=11):
    h = template.shape[0]
    w = template.shape[1]
    img = hr_image.copy()
    # All the 6 methods for comparison in a list
    #methods = ['cv.TM_SQDIFF_NORMED']
    methods = ['cv.TM_CCOEFF_NORMED']
    for meth in methods:
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 255, rectangle_thickness)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
    detected_spot_crop = hr_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if blur_filter == 'median':
        spot_contours=pipeline2(detected_spot_crop, median_blur_ksize=median_blur_ksize,contour_thickness=2, threshold_blocksize=threshold_blocksize)
    elif blur_filter == 'gaussian':
        spot_contours=pipeline3(detected_spot_crop, gblur_sigma=median_blur_ksize,contour_thickness=2, threshold_blocksize=threshold_blocksize)

    radii = []
    for spot_contour in spot_contours:
        (x,y),radius = cv.minEnclosingCircle(spot_contour)
        radii.append(radius)
    filtered_radii=[x for x in radii if x > radii_threshold]
    mean_spot_radii_px=np.mean(filtered_radii) * .6254

    return((mean_spot_radii_px, radii))

def get_scale_factor(triangle_dist_A, triangle_dist_B):
    sf = []
    for i in range(0,len(triangle_dist_A)):
        sf.append(triangle_dist_A[i]/triangle_dist_B[i])

    mean_sf = np.mean(sf)
    std_sf = np.std(sf, ddof=1)
    return(mean_sf,std_sf)

def triangle_dist(pairs):
    A = math.dist(pairs[0],pairs[1])
    B = math.dist(pairs[1],pairs[2])
    C = math.dist(pairs[0],pairs[2])
    return([A,B,C])

def read_10x_object(metadata, raw=False):
    # metadata is a pandas row
    folder_name = os.path.dirname(metadata['samples'])
    if raw == True:
        count_file = metadata['samples_raw']
    else:
        count_file = metadata['samples']
    adata = sc.read_10x_h5(filename = count_file)
    fields = metadata.index.tolist()
    fields.remove('samples')
    for field in fields:
        adata.obs[field] = [metadata[field]]*adata.n_obs
        adata.obs[field] = adata.obs[field].astype('category')

    adata.var['symbol'] = adata.var_names
    adata.var_names = adata.var['gene_ids']
    #pdb.set_trace()
    #adata.var['symbol'] = adata.var['symbol'].where(~adata.var['symbol'].duplicated(), adata.var['symbol'].index + '_1')
    adata.obs_names=[f"{x}-{metadata['sample_names']}" for x in adata.obs_names]
    adata.raw = adata
    return(adata)

def read_10x_object_table(infoTable):
    h5_objs = []
    for ind,row in infoTable.iterrows():
        adata = read_10x_object(row)
        h5_objs.append(adata)
    return(h5_objs)

def switch2symbol(h5_obj):
    if all(h5_obj.var_names == h5_obj.var['gene_ids']):
        h5_obj.var['gene_ids'] = h5_obj.var_names
        h5_obj.var_names = h5_obj.var['symbol']
    else:
        print('h5 already symbol')
    return(h5_obj)

def switch2ensembl(h5_obj):
    if all(h5_obj.var_names == h5_obj.var['symbol']):
        h5_obj.var['symbol'] = h5_obj.var_names
        h5_obj.var_names = h5_obj.var['gene_ids']
    else:
        print('h5 already ensembl')
    return(h5_obj)

def merge_h5s(h5_list):
    all_h5s = h5_list[0].copy()

    for i in range(1,len(h5_list)):
        all_h5s=anndata.concat([all_h5s,h5_list[i]],uns_merge="unique",join="inner",merge="same")
    return(all_h5s)
