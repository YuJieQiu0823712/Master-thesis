# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:27:17 2023

@author: Bart Steemans. Govers Lab.
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import tifffile as tiff
import numpy as np
import pickle
import cv2 as cv
import skimage
from scipy import interpolate
import math
from shapely.geometry import Polygon
from skimage.draw import polygon2mask
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import morphology as morph
from skimage import measure
from skimage.measure import regionprops_table, label
import sys
import logging
import time



px = 0.065


def load_imgs(files):
    imgs = []
    for file in files:
        img = tiff.imread(file)
        imgs.append(img)
    imgs = np.array(imgs)

    return imgs

def load_meshdata(mesh_file):
    with open(mesh_file, "rb") as file:
        mesh_dataframe = pickle.load(file)

    return mesh_dataframe
  
      
    
    

### RUN FUNCTIONS ONCE UNLESS THEY NEED TO BE RELOADED AFTER CHANGES
def crop_image(contour, image, pad = 10):
    contour = contour.copy()
    contour = contour.astype(int)
    mask = polygon2mask(image.shape, contour)

    
    padding = pad
    x, y, w, h = cv.boundingRect(contour)
    x = np.clip(x - padding, 0, image.shape[0])
    y = np.clip(y - padding, 0, image.shape[1])
    w = np.clip(w + 2 * padding, 0, image.shape[0] - x)
    h = np.clip(h + 2 * padding, 0, image.shape[1] - y) 
    
    
    cropped_img = image[x:x+w, y:y+h]
    cropped_mask = mask[x:x+w, y:y+h]
    cropped_mask = cropped_mask.astype(np.uint32)
    
   

        # Plot the cropped mask
    # plt.imshow(cropped_mask, cmap='gray')
    # plt.title('Cropped Mask')
    # plt.colorbar()
    # plt.show()
    

    
    return cropped_img, cropped_mask, x, y

def mesh2midline(x1s, y1s, x2s, y2s):
    
    line_list = []
 
    # if x1s, y1s, x2s, y2s == [] or x1s, y1s, x2s, y2s is None:
    #     print("There is no nucleoid mesh.") 
       
    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        line = np.array([x, y]).T
        line = smoothing_spline(line, n = len(x1), sf = 3, closed = False)
        line_list.append(line)
        
    return line_list

# def mesh2midline2(x1s, y1s, x2s, y2s):
    
#     cell_line_list = []
 
#     # if x1s, y1s, x2s, y2s == [] or x1s, y1s, x2s, y2s is None:
#     #     print("There is no nucleoid mesh.") 
       
   
#     x = (x1 + x2) / 2
#     y = (y1 + y2) / 2
#     line = np.array([x, y]).T
#     line = smoothing_spline(line, n = len(x1), sf = 3, closed = False)
#     cell_line_list.append(line)
        
#     return cell_line_list

def smoothing_spline(init_contour, n=200, sf=1, closed = True):
    if closed:
        tck, u = interpolate.splprep(init_contour.T, u=None, s=sf, per=1)
        #tck => A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.
        #u => An array of the values of the parameter.
    else:
        tck, u = interpolate.splprep(init_contour.T, u=None, s=sf)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = interpolate.splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T



#perform background correction on an input image. 
#It seems to calculate the average background value based on the provided image and parameters.
def phase_background(image, se_size=3, bgr_erode_num=1):
   
    # Crop the image to exclude borders
    cropped_image = image[int(image.shape[0]*0.05):int(image.shape[0]*0.95), int(image.shape[1]*0.05):int(image.shape[1]*0.95)]
    
    # Compute the threshold using Otsu's method. Separate pixels into two classes.
    thres = skimage.filters.threshold_otsu(cropped_image)
    
    # Create a binary mask using the threshold 
    mask = cv.threshold(image, thres, 255, cv.THRESH_BINARY_INV)[1]
    
    
    # Apply erosion to the mask to refine background
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (se_size, se_size))
    
    mask = cv.erode(mask, kernel, iterations=bgr_erode_num)
    
    # Convert mask to boolean
    bgr_mask = mask.astype(bool)
    
    # Calculate the mean of background pixels
    bgr = np.mean(image[bgr_mask])
    return bgr

#perform 2D interpolation on an input image using RectBivariateSpline
def interp2d(image):
    
    # Create a RectBivariateSpline interpolation function
    ff = interpolate.RectBivariateSpline(range(image.shape[0]), range(image.shape[1]), image, kx=1, ky=1)
      
    return ff




# def get_nuc_step_length(xs):
#     step_length_list=[]
    
    
#     #for x1, y1, x2, y2 in zip(nuc_x1, nuc_y1, nuc_x2, nuc_y2):
        
#     # calculated as the difference between corresponding x-coordinates of two line segments 
#     dx = x1[1:] + x2[1:] - x1[:-1] - x2[:-1]
#     #x1[1:] => start from index 1
    
#     dy = y1[1:] + y2[1:] - y1[:-1] - y2[:-1]
    
#     # Calculate the step length for each segment and multiply by pixel size
#     step_length = (np.sqrt(dx**2 + dy**2) / 2)*px
#     step_length_list.append(step_length)
    
#     return step_length_list



#calculate step lengths between points specified by (x1, y1) and (x2, y2)
def get_step_length(x1, y1, x2, y2, px):
    
    step_length_list=[]
    
    #for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
    
        
    # calculated as the difference between corresponding x-coordinates of two line segments 
    dx = x1[1:] + x2[1:] - x1[:-1] - x2[:-1]
    #x1[1:] => start from index 1
    
    dy = y1[1:] + y2[1:] - y1[:-1] - y2[:-1]
    
    # Calculate the step length for each segment and multiply by pixel size
    step_length = (np.sqrt(dx**2 + dy**2) / 2)*px
    step_length_list.append(step_length)

    return step_length_list
    
#calculate step lengths between points specified by (x1, y1) and (x2, y2)
def get_nucleoid_step_length(x1s, y1s, x2s, y2s, px):
    
    nuc_step_length_list=[]
    
    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
    
        
        # calculated as the difference between corresponding x-coordinates of two line segments 
        dx = x1[1:] + x2[1:] - x1[:-1] - x2[:-1]
        #x1[1:] => start from index 1
        
        dy = y1[1:] + y2[1:] - y1[:-1] - y2[:-1]
        
        # Calculate the step length for each segment and multiply by pixel size
        step_length = (np.sqrt(dx**2 + dy**2) / 2)*px
        nuc_step_length_list.append(step_length)

    return nuc_step_length_list    


#calculate step lengths between points specified by (x1, y1) and (x2, y2), 
#where the step length is computed for each segment and then multiplied by a pixel size (px). 
def get_length(step_length):
    return np.sum(step_length)


#calculate the average width of segments defined by coordinates (x1, y1) and (x2, y2)
def get_avg_width(x1s, y1s, x2s, y2s, px):
    
    width_not_ordered_list=[]
    width_list=[]
    
    
    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        
        #Calculate the width (distance) between corresponding points of each segment
        width_not_ordered = np.sqrt((x1-x2)**2+(y1-y2)**2) * px
        width_not_ordered_list.append(width_not_ordered)
        
        #Sort the widths in descending order
        sorted_width = sorted(width_not_ordered, reverse = True)
        
        #Calculate the average width using the top one-third of sorted widths
        width = (sum(sorted_width[:math.floor(len(sorted_width)/3)]) / math.floor(len(sorted_width)/3))
        width_list.append(width)
        
    average_width = [np.nan] if not width_list else np.mean(width_list)

    return width_list, width_not_ordered_list, average_width


#cell width variability calculated based on the 50% highest cell widths
#It takes a list of cell widths as input (width_no), sorts them in descending order, 
#and then calculates the coefficient of variation (CV) for the top half of the sorted widths.
# def get_cell_width_variability(width_no):
#     sorted_width = sorted(width_no, reverse = True)
    
#     half_idx = len(sorted_width) // 2
#     #use "//" to get a floored result of floating point division
   
#     half_width = sorted_width[:half_idx]
#     width_var = np.std(half_width) / np.mean(half_width)    
#     return width_var



#calculate the area enclosed by a given contour.
def get_area(contour, px):
    # Create a Polygon object from the given contour
    poly = Polygon(contour)
    
    # Calculate the area of the polygon
    area = poly.area
    
    # Scale the area by the square of the pixel size
    return area*px*px


# calculate the volume enclosed by a set of segments defined by (x1, y1) and (x2, y2) points.
def get_volume(x1, y1, x2, y2, step_length, px):
    
    # Calculate the distances between corresponding points of each segment
    d = np.sqrt((x1-x2)**2+(y1-y2)**2)
    
    # Calculate the cross-sectional areas of each segment
    volume = np.trapz((np.pi*(d/2)**2) , dx = step_length)
    # np.trapz => integrate by using the given step length.
    # Trapezoidal
    
    # Scale the volume by the square of the pixel size
    return volume*px*px


#calculate the surface area of a structure using given widths and a step length.
def get_surface_area(width_no, step_length):
    # Extract the widths excluding the first width
    # first width is a point
    widths = width_no[1:]
    
    # Calculate the surface areas for each segment
    surface_areas = 2 * np.pi * (widths / 2) * step_length
    #2 * np.pi * (widths / 2) => circumference
    
    # Calculate the total surface area by summing up individual surface areas
    total_surface_area = np.sum(surface_areas)
    return total_surface_area


#calculate the surface area-to-volume ratio. => not use this
def get_surface_area_over_volume(sa, vol):
    return (sa/vol)



#count the number of peaks in a given signal that satisfy certain conditions based on prominence and height. 
def find_signal_peaks(signal, maximum):
    
    # Find peaks in the signal based on prominence and height conditions
    peaks, _ = find_peaks(signal, prominence= 0.5, height = (maximum * 0.5))
    #prominence => Required prominence  of peaks. Either a number, None, an array matching x or a 2-element 
    #sequence of the former. The first element is always interpreted as the minimal and the second, 
    #if supplied, as the maximal required prominence.
    
    return len(peaks)


#calculate the kurtosis of a given signal.
def get_kurtosis(signals):
    
    data_list = []
    
    for signal in signals:
        data = kurtosis(signal)
        data_list.append(data)
    
    
    return data_list
   





#calculate the skewness of a given signal.
def get_skew(signals):
    
    data_list = []
    
    for signal in signals:
        data = skew(signal)
        data_list.append(data)
    
    return data_list



def mesh2contour(x1, y1, x2, y2):
    x2f = np.flip(x2)
    y2f = np.flip(y2)
    
    # Concatenate the x and y coordinates
    xspline = np.concatenate((x2f[1:], x1[1:]))
    yspline = np.concatenate((y2f[1:], y1[1:]))

    tck, u = interpolate.splprep(np.array([xspline, yspline]), k=3, s=2, per = 1) 
    u_new = np.linspace(u.min(), u.max(), 200)
    outx, outy = interpolate.splev(u_new, tck)
    
    return np.array([outx, outy]).T


#calculate a profile (or measurement) along a given midline using interpolation.
def measure_along_midline_interp2d(midlines, im_interp2d, width = 7, subpixel = 0.5):
    
    prf_list = []
    
    for midline in midlines:
        # Calculate a unit vector perpendicular to the midline
        unit_dxy = unit_perpendicular_vector(midline, closed = False)
        
        # Calculate a normalized vector for the width and subpixel resolution
        width_normalized_dxy = unit_dxy * subpixel
        #It prepares the vectors for use in generating profiles at varying distances from the midline.
        
        # Interpolate image values along the midline
        data = im_interp2d.ev(midline.T[0], midline.T[1])
        
        # Iterate over width steps and calculate profile values
        for i in range(1, 1+int(width * 0.5 / subpixel)):
            
            dxy = width_normalized_dxy * i
            #This vector represents a displacement in the direction perpendicular to the midline
            
            v1 = midline + dxy
            v2 = midline - dxy
            #calculate two points, v1 and v2, located at distances i units away from the midline in opposite directions.
            
            p1 = im_interp2d.ev(v1.T[0], v1.T[1])
            ##calculate a value p1 by evaluating the image data at the coordinates v1
            
            p2 = im_interp2d.ev(v2.T[0], v2.T[1])
            
            data = np.vstack([p1, data, p2])
            #Stack arrays in sequence vertically (row wise).
        
        # Calculate the average profile
        prf = np.average(data, axis=0)
        sigma = 2  # standard deviation of Gaussian filter
        
        prf = gaussian_filter1d(prf, sigma)
        #A Gaussian Filter is a low pass filter used for reducing noise (high frequency components) 
        #and blurring regions of an image.
        prf_list.append(prf)
    
    return prf_list


#calculate unit vectors that are perpendicular to a set of input data points. 
def unit_perpendicular_vector(data, closed= True):

    p1 = data[1:]
    p2 = data[:-1]
    dxy = p1 - p2 #directional components of the vectors.
    
    # Calculate angles and add π/2 to make the vector perpendicular
    ang = np.arctan2(dxy.T[1], dxy.T[0]) + 0.5 * np.pi
    #calculates the angles between the vectors and the x-axis using the arctan2 function
    #+ 0.5 * np.pi => makes the calculated angle 90 degrees (π/2) counterclockwise from the vector.
    
    #arctan => the function only receives the sign of x/y, and not x and y separately and cannot distinguish quadrants. 
    
    #if you are calculating something that ranges between -90 and 90 degrees like latitude, use arctan. 
    #If calculating an angle that can be between -180 and 180 degrees, use arctan2.
    
    dx, dy = np.cos(ang), np.sin(ang)
    
    # Create unit vectors
    unit_dxy = np.array([dx, dy]).T
    if not closed: 
        unit_dxy = np.concatenate([[unit_dxy[0]], unit_dxy])
        #[[unit_dxy[0]], unit_dxy] => creates a list of two arrays.
        #The first array contains only the first unit vector, and the second array contains all the unit vectors from unit_dxy.
        #concatenate => By duplicating the first element at the array, the path effectively becomes closed because the first and last elements are the same.
        
    else:
        unit_dxy = np.concatenate([unit_dxy,[unit_dxy[-1]]])
        
    return unit_dxy










############# Nucleoid ################# -------------------------------------------------------------------------------------------------------------------

# Unnecessary function as you can do this in the loop

def get_nucleoid_rect_length(nucleoid_contours, px):
    rect_length_list = []
    rect_width_list = []

    for nucleoid_contour in nucleoid_contours:
        nucleoid_contour = nucleoid_contour.astype(np.float32)
        
        x_y, width_height, angle_of_rotation = cv.minAreaRect(nucleoid_contour)
        
        length = max(width_height) * px
        rect_length_list.append(length)
        width = min(width_height) * px
        rect_width_list.append(width)
        
    total_rect_length = sum(rect_length_list)
    # Calculate average width, handling division by zero
    average_rect_width = [np.nan] if not rect_width_list else np.mean(rect_width_list)
    
    return rect_length_list, rect_width_list, total_rect_length, average_rect_width



# In the application we might be able to do something with the nucleoid masks. For now, however, it is not necessary to implement it.
#skeleton and distance transform
def get_nucleoid_avg_width(nucleoid_masks, px):
    
    nucleoid_avg_width_list = []
    width_not_ordered_list = []

    
    for nucleoid_mask in nucleoid_masks:
      
        
        # Convert nucleoid_mask to 8-bit unsigned integer type
        nucleoid_mask = nucleoid_mask.astype(np.uint8)

        # Ensure both arrays have the same data type
        distance = cv.distanceTransform(nucleoid_mask, distanceType=cv.DIST_L2, maskSize=5).astype(np.float64)
        skeleton = morph.skeletonize(nucleoid_mask).astype(bool)
        
        # Extract values at skeleton locations
        width = distance[skeleton] * 2 * px
        width_not_ordered_list.append(width)
        nucleoid_avg_width_list.append(np.mean(width))
   
 
        
        
    return nucleoid_avg_width_list, width_not_ordered_list

def get_nucleoid_mask(nucleoid_contours,cropped_image,cell_x_offset,cell_y_offset):
    
    nucleoid_mask_list=[]
    
    for nucleoid_contour in nucleoid_contours:
        nucleoid_contour = nucleoid_contour.copy()
        nucleoid_contour.T[1] = nucleoid_contour.T[1]-cell_y_offset
        nucleoid_contour.T[0] = nucleoid_contour.T[0]-cell_x_offset
        cropped_mask = polygon2mask(cropped_image.shape, nucleoid_contour)
        cropped_mask = cropped_mask.astype(int)
        nucleoid_mask_list.append(cropped_mask)
        
    return nucleoid_mask_list


def get_nucleoid_aspect_ratio(rect_length_lists, rect_width_lists):
    
    aspect_ratio_list=[]

    for length_list, width_list in zip(rect_length_lists, rect_width_lists):
        aspect_ratio = width_list / length_list
        aspect_ratio_list.append(aspect_ratio)
    avg_aspect_ratio = [np.nan] if not aspect_ratio_list else np.mean(aspect_ratio_list)

    return aspect_ratio_list,avg_aspect_ratio

#calculate the area enclosed by a given contour.
def get_nucleoid_area(contours, px):
    
    area_list=[]
    for contour in contours:
        
      poly = Polygon(contour)
      area = np.array((poly.area)*px*px)
      area_list.append(area)
    
    total_area = np.sum(area_list)
    return area_list, total_area

def get_nucleoid_curvature_characteristics(contours, px):
    max_c_list = []
    min_c_list = []
    mean_c_list = []
    std_c_list = []

    for contour in contours:
        # Calculate gradients of x and y coordinates
        dx = np.gradient(contour[:, 0] * px)
        dy = np.gradient(contour[:, 1] * px)

        # Calculate second derivatives of gradients
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # Compute curvature using the formula
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5

        max_c = np.nanmax(curvature)
        min_c = np.nanmin(curvature)
        mean_c = np.nanmean(curvature)
        std_c = np.std(curvature)

        max_c_list.append(max_c)
        min_c_list.append(min_c)
        mean_c_list.append(mean_c)
        std_c_list.append(std_c)
        
    avg_mean_c = [np.nan] if not mean_c_list else np.mean(mean_c_list)

    return max_c_list, min_c_list, mean_c_list, std_c_list,avg_mean_c

def get_nucleoid_perimeter_measurements(contours, areas, px):
    perimeter_list = []
    circularity_list = []
    compactness_list = []
    sphericity_list = []
    
    for contour, area in zip(contours, areas):
        d = np.diff(contour, axis=0)
        # Calculate the Euclidean distances for all consecutive points in the contour.
        distances = np.sqrt(np.sum(d**2, axis=1))
        perimeter = np.sum(distances) * px
        
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter**2)
        
        # Calculate compactness
        compactness = (perimeter ** 2) / area
        
        # Calculate sphericity
        sphericity = (np.pi * 1.5 * ((perimeter / (2 * np.pi)) ** 1.5)) / area
        
        perimeter_list.append(perimeter)
        circularity_list.append(circularity)
        compactness_list.append(compactness)
        sphericity_list.append(sphericity)
        
    avg_perimeter = [np.nan] if not perimeter_list else np.mean(perimeter_list)
    avg_circularity = [np.nan] if not circularity_list else np.mean(circularity_list)
    avg_compactness = [np.nan] if not compactness_list else np.mean(compactness_list)
    avg_sphericity = [np.nan] if not sphericity_list else np.mean(sphericity_list)


    return perimeter_list, circularity_list, compactness_list, sphericity_list,avg_perimeter,avg_circularity,avg_compactness,avg_sphericity

# calculate the nucleoid volume by ratio of nucleoid area and cell column
def get_nucleoid_volume(cell_volume,nucleoid_areas,cell_area):
    
    nucleoid_volume_list=[]
    
    for nucleoid_area in nucleoid_areas:
        #The power 3/2 was used to convert the estimated nucleoid area fraction into a volume fraction.
        
        nucleoid_volume = cell_volume * (nucleoid_area/cell_area) ** (3/2)
        #https://pdf.sciencedirectassets.com/272196/1-s2.0-S0092867420X00158/1-s2.0-S0092867421006899/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEB0aCXVzLWVhc3QtMSJIMEYCIQCrpOJzep9aPAmu8i%2BMmj7tIMRfSjrPleF4JUwbOmVKbQIhAPor7D%2FB6tQsE3rSrVdGJO6dnNK1rzbkkM%2F2YaW3nUzeKrMFCBUQBRoMMDU5MDAzNTQ2ODY1Igw%2FfgA%2FjSyH75fB2UYqkAXf5zL8HQIUYqzALzxExa3vWGzBBKj2IEiQekxor8f30cPn88wCAyISAuQi0jcHh3EndiTARSdT9AZs40IsS0UuIAJi06%2BWuWFGV2UBDRdq4I4zblTRnFVJ7TLRmRcoC8dmpGy4XDScFiw99xkRX4yAvlDpmDm7vkeOQ3BO5jSAqxKH0wID017eOlyOsoWvkyr7tZruNIxQZDeGBxHEmuKBZHX3WgaCzHl9o2acJuuQ7IxBu7Wz%2BQWnw0lg79bJAdI61yelAd0h8HlqQpy1%2BmuSU80nf%2FDtN6nzOyOFYtkOFxnTbij4FE95JHAkZNf4rT6M4g2OxVck3q3dIWk9sXdyh%2FzVPRFBHGTxBGtEHrnxVEzXklrR5JZ6tpKF0c0ljWiXdp8yPBAcwk70vKIiYqsvkkjkSkJn9kZrY%2BLu%2F2uemKdhjCBqzlXcCRVtRinW157zmibxQVBHXD8Liq%2F3sPEmPHNWc%2FQfWssztY5JIqf7qjwnf9XdaUTjeztos1dXwOPGZfIvFNcGQGjIszF2Q4LlKQmSRnRUeJSbynDdbBmAQwdaJlCeQuYNUz4nGqSyC1hTeaZdj3jsLEpQi7vbSDCxA2pPDGj%2Ff48DJ4drMsQ1s6u6E%2B4OerOcHVlU9gZ%2B0qW3UcZ3k1kLxoImLnuqSNJA9R%2Bv9%2FZNDLBHcWQNQmOG5GjHo1zMxyLi7Ly0HZZ4HxiVrg6dpyz4Da%2FcwshE8hvvTKibD6U6zhmCvWWxhkm5SjegLP%2FEelF8Plf2vvTzUIwnkVEeu8tEWINa6kz3XMB51EWR7g7X5w19xDfbLz%2FpmaC21xCnqwTJt2X0z7Qo3YJoXQcDz3oSoIwQihpMIV809aKtMHPCZbRH0UouWoFdrTDLi7aoBjqwAYSoqOdLa%2FZlCWQmPULssM64TFzKQJkJcC%2FiyOX30yEENPveT1vJU6sM2WPUkHYHF7T3oNnwjK4dqtskdprtmYC1N1aH6gj0RCFeWsiEoO%2F0c6vRngctfV6C%2BipWphtxbLQC465IyqIeWDCMhbSSeX4dvUId2K%2BIXg2%2Bo%2ByjC%2BzBn%2BNU5gJWZH%2BgZsmDKTN6CoTe44OEUwo1rm4j7psaWHJXVF1pOKVH37iWb9OE6ZjB&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230922T125122Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTIPTOYMX%2F20230922%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8b9f9f49fdafa60983bb45eaf045cad7cf74464fb78443ea09c6b50c6b1a7ad9&hash=bd877942991072372ba979aade73e4f9c259fa95fc4c3c63d5207d8a86957044&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0092867421006899&tid=spdf-d6c3c8bc-a416-471a-9ea8-d972515304eb&sid=56d275e37f769446e71a5bb0d772a9141894gxrqb&type=client&tsoh=d3d3LXNjaWVuY2VkaXJlY3QtY29tLmt1bGV1dmVuLmUtYnJvbm5lbi5iZQ%3D%3D&ua=18035853050f060a06035b&rr=80aaadcd982ff965&cc=be
        
        nucleoid_volume_list.append(nucleoid_volume)
        total_volume = np.sum(nucleoid_volume_list)
    
    
        #cell_volume has mutiply px*px, so don't need to mutiply px*px
    return nucleoid_volume_list,total_volume

#cell width variability calculated based on the cell widths
#It takes a list of cell widths as input, sorts them in descending order, 
#and then calculates the coefficient of variation (CV) for the sorted widths.
def get_nucleoid_width_variability(widths):
    #Because the CV is unitless and usually expressed as a percentage, it is used instead of the SD to compare the spread of data 
    # sets that have different units of measurements or have the same units of measurements but differs greatly in magnitude.
    #Ratio between the standard deviation and the mean ( which in some case could be negative)
 
    width_var_list=[]
    if widths:
        
        for width in widths:         
            width_var = np.std(width) / np.mean(width)       
            width_var_list.append(width_var)
       
    return width_var_list


#width_divided_by_largest_width
def track_nucleoid_width_variability(widths_not_ordered):
    
    track_width_variability_list=[]

    for width_not_ordered in widths_not_ordered:
        if len(width_not_ordered) > 0:
            width_variability = max(width_not_ordered) / width_not_ordered
            track_width_variability_list.append(width_variability)
        else:
            # Handle the case when the input list is empty
            track_width_variability_list.append([]) 
   
    return track_width_variability_list


def get_object_intensity(nucleoid_masks, im_interp2d, x_offset, y_offset):
    
    
    
    total_intensity = []
    max_intensity = []
    mean_intensity = []
    
    for mask in nucleoid_masks:
        
        coords = np.column_stack(np.where(mask)).astype(int)
        if len(coords) == 0:
            total_intensity.append([])  # Default value if the list is empty
            max_intensity.append([])
            mean_intensity.append([])
        else:
            values = im_interp2d.ev(coords[:, 0] + x_offset, coords[:, 1] + y_offset)
            total_intensity.append(np.sum(values))
            max_intensity.append(np.max(values))
            mean_intensity.append(np.mean(values))
    
    
    
    avg_mean_intensity = [np.nan] if not mean_intensity else np.mean(mean_intensity)
    
    return total_intensity, max_intensity, mean_intensity,avg_mean_intensity



#This method return convexity, eccentricity, and solidity
def get_object_convexity(cropped_masks):
   
    convexity_list_test = []
    eccentricity_list_test=[]
    solidity_list_test=[]
    
    for nucleoid_mask in cropped_masks:
       
        labels = measure.label(nucleoid_mask)
        
        properties = regionprops_table(labels, properties=('perimeter', 'eccentricity', 'solidity','convex_image' ))
        properties_df = pd.DataFrame(properties)
        
        image_perimeter = properties_df.loc[0,'perimeter']
        image_eccentricity = properties_df.loc[0,'eccentricity']
        image_solidity = properties_df.loc[0,'solidity']
        convex_image = properties_df.loc[0,'convex_image']
    
        
        convex_image_perimeter = measure.regionprops(convex_image*1)[0].perimeter
        convexity = round(convex_image_perimeter/image_perimeter, 3)
        
        convexity_list_test.append(convexity)
        eccentricity_list_test.append(image_eccentricity)
        solidity_list_test.append(image_solidity)
    
    return convexity_list_test, eccentricity_list_test, solidity_list_test




def get_nucleoid_mesh_coor(nucleoid_meshs):
    
    x1_list=[]
    y1_list=[]
    x2_list=[]
    y2_list=[]
    
    
    
    if nucleoid_meshs == [] or nucleoid_meshs is None:
        print("There is no nucleoid mesh.") 
       

    else:
        for nucleoid_mesh in nucleoid_meshs:
      
            x1 = nucleoid_mesh[:,0]
            y1 = nucleoid_mesh[:,1]
            x2 = nucleoid_mesh[:,2]
            y2 = nucleoid_mesh[:,3]
            
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
    
    return x1_list,  y1_list, x2_list, y2_list
    
# calculates the constriction degree in relative and absolute value, 
# and the relative position along the length of the cell
# absolute constr degree needs to be multiplied by the width
def constr_degree_single_cell_min(intensitys, new_step_lengths):
    
    constrDegree_list=[]
    relPos_list=[]
    constrDegree_abs_list=[]
    ctpos_list=[]
    
    for intensity,new_step_length in zip(intensitys,new_step_lengths):
    

        minima = np.concatenate(([False], (intensity.T[1:-1] < intensity.T[:-2]) & (intensity.T[1:-1] < intensity.T[2:]), [False]))
        
          # checks if the intensity list is empty or if the sum of the intensity values is zero.
        if all(not x for x in minima) or sum(intensity.T) == 0: 
       
            minsize = 0
            minsizeabs = 0
            ctpos = []
            
            # constrDegree_list.append(minsize)
            # constrDegree_abs_list.append(minsizeabs)
            # ctpos_list.append(ctpos)
       
        else:
            #identifies the local minima in the intensity values.
            index = np.where(minima)[0]
    
            dh = np.zeros(index.shape) 
            dhi = np.zeros(index.shape)
            hgt = np.zeros(index.shape)
            for i in range(len(index)):
                k = index[i]
                half1 = intensity.T[:k-1]
                half2 = intensity.T[k+1:]
                try:
                    dh1 = np.max(half1)-intensity.T[k]
                    dh2 = np.max(half2)-intensity.T[k]
                    dh[i] = np.min([dh1, dh2]) #difference in height (dh) between the local minima and the maximum values in the two halves of the intensity values.
                    dhi[i] = np.mean([dh1, dh2]) #calculates the average difference in height (dhi) and the height (hgt) of the local minima.
                    hgt[i] = intensity.T[k]+dhi[i]
                except ValueError:
                    minsize = 0
                    minsizeabs = 0
                    ctposr = []
                    
                    # constrDegree_list.append(minsize)
                    # constrDegree_abs_list.append(minsizeabs)
                    # relPos_list.append(ctposr)
    
                #finds the index of the local minimum with the largest difference in height (dh)
                #and assigns the corresponding values of dhi and hgt to minsizeabs and minsize, respectively.
                fix = np.argmax(dh) 
                minsizeabs = dhi[fix] #absolute constriction degree (minsizeabs)
                minsize = minsizeabs / hgt[fix] #the constriction degree (minsize)
    
                #calculates the relative position along the length of the cell (ctposr) based on the cumulative sum of the new_step_length values.
                ctpos = index[fix] #the index of the local minimum (ctpos)
                ctpos_list.append(ctpos)
                ctposr = np.cumsum(new_step_length)[ctpos] / np.sum(new_step_length)
            if not minsize:
                minsize = 0
                minsizeabs = 0
                # constrDegree_list.append(minsize)
                # constrDegree_abs_list.append(minsizeabs)
                
        constrDegree = minsize
        constrDegree_abs = minsizeabs
        constrDegree_list.append(constrDegree)
        constrDegree_abs_list.append(constrDegree_abs)
       
        if not ctpos:
            ctposr = np.nan
            ctpos_list.append(ctpos)

        relPos = ctposr #relative position (relPos)
        relPos_list.append(relPos)
    consDegree  = [np.nan] if not  constrDegree_list else np.mean(constrDegree_list)  
    return constrDegree_list, relPos_list, constrDegree_abs_list, ctpos_list
    



def bend_angle(object_contours, window=10):
    
    for object_contour in object_contours:
        p1 = np.concatenate((object_contour[-window:],object_contour[:-window])).T
        p2 = object_contour.copy().T
        p3 = np.concatenate((object_contour[window:],object_contour[0:window])).T
        p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
        p1p3 = p1[0]*1+p1[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p1p3/p1p2, deg=True)





def generate_feature_frame(mesh_dataframe,imgs):
    # RUN THIS PART SEPARATELY WHEN YOU WANT TO TEST THE CODE ----------------------------------------------------------------------------------------------
    dataframe = mesh_dataframe.copy()
    # You have to create the column beforehand if you want to assign lists of varying lengths to the column
    columns_to_assign_none = ['object_avg_width_per_nucleoid', 
                              'object_width_variability', 
                              'object_aspect_ratio', 
                              'object_areas', 
                              'max_curvature',
                              'min_curvature',
                              'mean_curvature',
                              'std_curvature',
                              'object_perimeters',
                              'object_circularities',
                              'object_compactnesses',
                              'object_sphericities',
                              'object_convexity',
                              'object_eccentricity',
                              'object_solidity',
                              'object_track_width_vars',
                              'object_volumes',
                              'object_total_intensity',
                              'object_max_intensity',
                              'object_mean_intensity',
                              'midline_skewness',
                              'midline_kurtosis',
                              'object_constriction_degree',
                              'object_relative_position']
    
    dataframe[columns_to_assign_none] = None
    
    #for frame_num in range(0, len(imgs)):
    for frame_num in range(0, 1): 
    
        frame_df = dataframe[dataframe['frame'] == frame_num]
        
        image = imgs[frame_num]
        
        im_interp2d = interp2d(image)
    
        
        for index, row in tqdm(frame_df.iterrows()):
           
    
            cell_id = row['cell_ids']
            mesh = row['mesh']
            
            
            nucleoid_contours = row['object_contours']
            nucleoid_mesh = row['object_mesh']
            # nucleoid_skeleton = row['object_skeleton']
    
            
            x1 = mesh[:,0]
            y1 = mesh[:,1]
            x2 = mesh[:,2]
            y2 = mesh[:,3]
            
            
            if 4 <= len(x1) <= 500 and nucleoid_mesh != [] and nucleoid_mesh is not None:
                try:
                    step_length = get_step_length(x1, y1, x2, y2, px)
                    contour = mesh2contour(x1, y1, x2, y2)
                    cell_cropped_img, cell_cropped_mask, cell_x_offset, cell_y_offset = crop_image(contour, image)
                    cell_volume = get_volume(x1, y1, x2, y2, step_length, px)
                    
                    nuc_x1, nuc_y1, nuc_x2, nuc_y2 = get_nucleoid_mesh_coor(nucleoid_mesh)
                    nuc_step_length = get_nucleoid_step_length(nuc_x1, nuc_y1, nuc_x2, nuc_y2, px)
                    nucleoid_masks = get_nucleoid_mask(nucleoid_contours,cell_cropped_img,cell_x_offset,cell_y_offset)
                   
                    
                    # Number of nucleoid features
                    dataframe.loc[index, 'object_number'] = len(nucleoid_contours)
                    rect_length_list, rect_width_list, dataframe.loc[index, 'object_total_rect_length'],dataframe.loc[index, 'object_total_rect_width']  = get_nucleoid_rect_length(nucleoid_contours, px)
                    dataframe.at[index,'object_avg_width_per_nucleoid'], width_not_oredered, dataframe.loc[index,'object_avg_width'] = get_avg_width(nuc_x1, nuc_y1, nuc_x2, nuc_y2, px) #use top one-third sorted widths
                    
                    avg_width, width_no = get_nucleoid_avg_width(nucleoid_masks, px) #distanceTransform            
                    dataframe.at[index, 'object_width_variability'] = get_nucleoid_width_variability(width_no)
                    dataframe.at[index, 'object_aspect_ratio'], dataframe.loc[index, 'object_avg_aspect_ratio'] = get_nucleoid_aspect_ratio(rect_length_list, rect_width_list)
                    dataframe.at[index, 'object_areas'], dataframe.loc[index, 'object_total_area'] = get_nucleoid_area(nucleoid_contours, px)
                    
                    max_c, min_c, mean_c, std_c,dataframe.loc[index, 'object_avg_curvature'] = get_nucleoid_curvature_characteristics(nucleoid_contours, px)
                    dataframe.at[index, 'max_curvature'] = max_c
                    dataframe.at[index, 'min_curvature'] = min_c
                    dataframe.at[index, 'mean_curvature'] = mean_c
                    dataframe.at[index, 'std_curvature'] = std_c
        
                    
                    perimeters, circularities, compactnesses, sphericities,avg_peri,avg_cir,avg_comp,avg_sph = get_nucleoid_perimeter_measurements(nucleoid_contours, dataframe.at[index, 'object_areas'], px)
                    dataframe.at[index, 'object_perimeters'] = perimeters
                    dataframe.at[index, 'object_circularities'] = circularities
                    dataframe.at[index, 'object_compactnesses'] = compactnesses
                    dataframe.at[index, 'object_sphericities'] =  sphericities
                    dataframe.at[index, 'object_avg_perimeters'] = avg_peri
                    dataframe.at[index, 'object_avg_circularities'] = avg_cir
                    dataframe.at[index, 'object_avg_compactnesses'] = avg_comp
                    dataframe.at[index, 'object_avg_sphericities'] =  avg_sph                
                    
                    ####
                    dataframe.at[index,'object_convexity'],dataframe.at[index,'object_eccentricity'],dataframe.at[index,'object_solidity']  =  get_object_convexity(nucleoid_masks)
       
                    
                    track_widths = track_nucleoid_width_variability(width_no)
                    dataframe.at[index, 'object_track_width_vars'] = get_nucleoid_width_variability(track_widths)
                    
                    dataframe.at[index, 'object_volumes'],dataframe.at[index, 'object_total_volume'] = get_nucleoid_volume(cell_volume, dataframe.at[index, 'object_areas'], get_area(contour, px))
                    # get_surface_area(width_not_ordered, step_length)
                    # get_surface_area_over_volume(sa, vol)
                    
                    
                    
                    
                    # Intensity
                    total_intensity, max_intensity, mean_intensity, dataframe.loc[index, 'object_avg_intensity'] = get_object_intensity(nucleoid_masks, im_interp2d, cell_x_offset, cell_y_offset)
                    dataframe.at[index, 'object_total_intensity'] = total_intensity
                    dataframe.at[index, 'object_max_intensity'] = max_intensity
                    dataframe.at[index, 'object_mean_intensity'] = mean_intensity
                  
                    
                    # Midline intensity features              
                    midline = mesh2midline(nuc_x1, nuc_y1, nuc_x2, nuc_y2)       
                    axial_intensity = measure_along_midline_interp2d(midline, im_interp2d, width=5)
                    dataframe.at[index, 'midline_skewness'] = get_skew(axial_intensity) #negative => left tail
                    dataframe.at[index, 'midline_kurtosis'] = get_kurtosis(axial_intensity) #negative => light tails
                    
                    
                    
                    # constriction
                    constrDegree, relPos, constrDegree_abs, ctpos = constr_degree_single_cell_min(axial_intensity,nuc_step_length)
                    dataframe.at[index, 'object_constriction_degree'] = constrDegree
                    dataframe.at[index, 'object_relative_position'] = relPos
                    
                    
            
                
                
                except Exception:
                    print(f'Error for cell id: {cell_id} and frame {frame_num}')
            else:
                print(f'The mesh of Cell/nucleoid {cell_id} is too small/big ')

    return dataframe


#pre-processing features into single digit
def preprocessing_dataframe(dataframe):
    
    preprocessing_dataframe = dataframe.copy()
    preprocessing_dataframe = preprocessing_dataframe.dropna() #drop entire row with Na value
    preprocessing_dataframe = preprocessing_dataframe.iloc[:,-13:] # keep the last 13 columes
    preprocessing_dataframe.isna().sum() #check no missing value in dataframe
    
   
    return preprocessing_dataframe




def save_data(frameName,outputfile):
    frameName.to_pickle(outputfile)


if __name__ == "__main__":

    input_mesh = sys.argv[1]
    input_imgs = sys.argv[2:-1] 
    output_file = sys.argv[-1]
    
    mesh_dataframe = load_meshdata(input_mesh)
    imgs = load_imgs(input_imgs)
    dataframe = generate_feature_frame(mesh_dataframe,imgs)
    processed_dataframe = preprocessing_dataframe(dataframe)
    save_data(processed_dataframe,output_file)
    
 








# ###### Standard Scaling (mean = 0 and variance = 1)
# from sklearn import datasets
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# #PCA preserves the variance in the data, whereas t-SNE preserves the relationships 
# #between data points in a lower-dimensional space


# #standardize the feature matrix
# X= StandardScaler().fit_transform(processed_dataframe)
# print(X.shape)
# print(X)
    
# #create a PCA that will retain 85% of the variance
# pca = PCA(n_components=0.85,whiten=True)

# #conduct PCA 
# X_pca = pca.fit_transform(X)
# print(X_pca.shape)
# print(X_pca) #n_component=3

# #create a PCA with 2 components
# pca = PCA(n_components=2,whiten=True)
# X_pca = pca.fit_transform(X)
# print(X_pca.shape)
# print(X_pca)

# pca_df = pd.DataFrame(data = X_pca, columns = ['PC1','PC2'])
# pca_df #397x2

# # Visualization
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize = (10,6))
# c_map = plt.cm.get_cmap('jet', 10)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], s = 15, cmap = c_map)
# plt.xlabel('PC-1') , plt.ylabel('PC-2')
# plt.show()






# ### t-SNE = non-linear ###
# from sklearn.manifold import TSNE

# pca_score= PCA().fit_transform(processed_dataframe)
# pca_df = pd.DataFrame(pca_score)

# #By default, TSNE() function uses the Barnes-Hut approximation, which is computationally less intensive.
# tsne = TSNE(n_components = 2, perplexity = 4, early_exaggeration = 12, 
#                 n_iter = 1000, learning_rate = 33, verbose = 1).fit_transform(pca_df.loc[:,0:12]) #13 features
#perplexity is the most important parameter in t-SNE, and it measures the effective number of neighbors.
#(standard range 10-100)
#In case of large, datasets, keeping large perplexity parameter (n/100; where n is the number of observations) is helpful for preserving the global geometry.
#In addition to the perplexity parameter, other parameters such as the number of iterations (n_iter), 
#learning rate (set n/12 or 200 whichever is greater), and early exaggeration factor (early_exaggeration) 
#can also affect the visualization and should be optimized for larger datasets (Kobak et al., 2019).
#https://www.reneshbedre.com/blog/tsne.html
#https://www.nature.com/articles/s41467-019-13056-x


# create dataframe
# cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
#                        data=np.column_stack((tsne, 
#                                             target.iloc[:30000])))
# # cast targets column to int
# cps_df.loc[:, 'target'] = cps_df.target.astype(int)
# cps_df.head()




# # plot t-SNE clusters
# from bioinfokit.visuz import cluster

# tsne_plot = cluster.tsneplot(score=tsne)
# #plot will be saved in same directory (tsne_2d.png) 
# # Plot t-SNE clusters without saving to a file

# # Display the plot using matplotlib's pyplot
# plt.figure(figsize=(8, 6))
# plt.scatter(tsne_plot[:, 0], tsne_plot[:, 1])
# plt.xlabel('tsne-1') , plt.ylabel('tsne-2')
# plt.title("t-SNE Clusters")
# plt.show()





# UPGMA

# UMAP




