import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from tensorflow import keras
from PIL import Image 
import imageio
import skimage.io
import math
from scipy.ndimage.morphology import binary_closing, binary_dilation

def read_video(path):
    cap = cv2.VideoCapture(path)
    video_frames = []
    ret, frame = cap.read()
    while ret :
        video_frames.append(frame)
        ret, frame = cap.read()
    return np.array(video_frames)

def detect_arrow(img):
    w,h,c = img.shape
    arrow = np.zeros([w,h])
    stds = np.std(img,axis = 2)
    arrow[(np.argmax(img,axis=2) == 2) * (stds > 10)] = 255
    return arrow

def compute_path(video_frame):
    path = []
    for img in video_frames:
        arrow = detect_arrow(img)
        cy,cx = ndimage.measurements.center_of_mass(arrow)
        path.append((int(cx),int(cy)))
    return path

def low_pass_filter(img,size):
    '''
    Low-pass filter applied to the simple images in order to get rid of noise.
    
    Input: grayscale image
    Output: filtered image
    
    '''
    Kernel = np.ones((size,size),np.float32)/size**2
    return ndimage.filters.convolve(img, Kernel)

def binaryImage(im):
    '''
    Threshold function that returns values of 0 or 255 if image pixel values are within thresholds
    
    Input: image
    Output: 0 or 255 'binary' image
    '''
    im_filt = cv2.medianBlur(im,1)
    im_binary = cv2.inRange(im_filt,100,255) #threshold values found from 
    return im_binary

def initial_rectangle(im, param):
    [h,w] = im.shape
    region = []
    corner = [] 
    
    for i in range(h):
        for j in range(w):
            if im[i,j] == 255:
                region.append([i,j])
    region = np.array(region)
    
    max_x = np.max(region[:,1])
    min_x = np.min(region[:,1])
    max_y = np.max(region[:,0])
    min_y = np.min(region[:,0])  
    
    left = min_x-param
    top = min_y-param
    right = max_x+param
    bottom = max_y+param
    
    pt1, pt2 = (right, bottom), (left, top)
    pt3, pt4 = (right, top), (left, bottom)
    
    corner = [left, right, top, bottom]
    im_out = np.copy(im)
    im_out = cv2.rectangle(im_out, pt1, pt2, 255, 2)
    
    return im_out, corner

def symbol_detect(im) :
    '''
    Function that takes input image and processes it to obtain large binary "blobs"
    around each symbol that needs to be evaluated
    
    Input : First Frame of video
    Output : Mask of all symbols of first frame, minus the arrow
    '''
    im_arrow = detect_arrow(im)
    im_r_a, value = initial_rectangle(im_arrow,20)       
    im_bin = binaryImage(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    
    for i in range (value[2],value[3]):
        for j in range (value[0], value[1]):
            im_bin[i,j] = 255
    im_close = binary_closing(im_bin)
    im_dilate = binary_dilation(np.logical_not(im_close))
    im_filt = low_pass_filter(np.logical_not(im_dilate),9)
    return np.logical_not(im_filt).astype(int)*255

def get_unvisited_neighbors(img,xy,img_label, threshold):
    # Return unvisited neighbors
    (x,y) = xy
    (w,h) = img.shape
    neighbors = []
    
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if i>=0 and i<w and j>=0 and j<h:
                
                # if the neighbor pixel satisfies the threshold and hasn't been visited yet
                if img_label[i,j]==0 and img[i,j]>threshold:
                    neighbors.append((i,j))
                    
    return neighbors

def region_growing(img, seed, threshold):
    # Return a region starting from a pixel for a given threshold
    
    pixel_labels = np.zeros(img.shape)
    stack = []
    stack.append(seed)
    
    while len(stack)>0:
        pixel_to_label = stack.pop()
        pixel_labels[pixel_to_label] = 1
        neighbors = get_unvisited_neighbors(img,pixel_to_label,pixel_labels, threshold)
        
        for i in range(len(neighbors)):
            stack.append(neighbors[i])
            
            # pixel is labelled
            pixel_labels[neighbors[i]]=1
            
    return pixel_labels

def shape_extraction(im,threshold=0.5):
    mask = np.copy(im)
    (w,h) = mask.shape
    regions = []
    for i in range(w):
        for j in range(h):
            if mask[i,j] > threshold :
                shape = region_growing(mask,(i,j),threshold)
                mask[shape > threshold] = 0
                nb_w_pxl = cv2.countNonZero(shape)
                if  nb_w_pxl < 1000 and nb_w_pxl > 200 : # threshold to get rid of the larger and smaller blobs
                    regions.append(shape)
                    
    return regions

def create_object_image(mask_array, im, param):
    '''
        mask_array is an array of all masks
        im is original first frame
        param is parameter for extra width around numbers
        
        output is an array of all the cropped symbols    
    '''
    cropped_im = []
    full_im = []
    cropped_cm = []
    for k in range (0,len(mask_array)) :
        mask = mask_array[k]    
        [h,w] = mask.shape
        region = []
        corner = [] 

        for i in range(h):
            for j in range(w):
                if mask[i,j] != 0:
                    region.append([i,j])
        region = np.array(region)

        cy, cx = ndimage.measurements.center_of_mass(mask)
        cm = (int(cx), int(cy))        

        max_x = np.max(region[:,1])
        min_x = np.min(region[:,1])
        max_y = np.max(region[:,0])
        min_y = np.min(region[:,0]) 

        left = min_x-param
        top = min_y-param
        right = max_x+param
        bottom = max_y+param
        width, hight = right-left, bottom-top
        if width > hight : # needed to make images square
            diff = width - hight
            left,right = left + diff/2, right - diff/2
        else : 
            diff = hight - width
            top, bottom = top + diff/2, bottom - diff/2
        
        crop = im[int(top):int(bottom), int(left):int(right)]
        full_im.append(crop)
        cropped_im.append(cv2.resize(crop,dsize=(28,28),interpolation = cv2.INTER_NEAREST))
        cropped_cm.append(cm)
    
    return cropped_im, full_im ,cropped_cm

def features_extraction(symb_array):
    bin_arr = []
    for i in range (0,len(symb_array)):
        im_gray = cv2.cvtColor(np.max(symb_array[i])-symb_array[i],cv2.COLOR_BGR2GRAY)
        im_gray = im_gray-np.min(im_gray)
        im_gray = cv2.GaussianBlur(im_gray,(3,3),cv2.BORDER_CONSTANT)
        bin_arr.append(im_gray)
    return bin_arr

def close(cm1,cm2):
    threshold = 20
    x = np.abs(cm1[0]-cm2[0])
    y = np.abs(cm1[1]-cm2[1])
    return x < threshold and y < threshold

def draw_path(image,path,w,h,nframe):
    a = path[:nframe]
    for point1, point2 in zip(a, a[1:]): 
        cv2.line(image, point1, point2, [50, 50, 180], 4) 
    return image

def initiate_symbol_detection():
    image_operators = cv2.imread("original_operators.png", cv2.IMREAD_GRAYSCALE)
    op_h, op_w = image_operators.shape
    step = int(op_w/5)
    im_operator = []
    for i in range(0, op_w-step, step):
        im_sub = image_operators[:,i+15:i+op_h+15]
        im_sub_28 = cv2.resize(im_sub,dsize=(28,28),interpolation = cv2.INTER_LINEAR)
        im_operator.append(im_sub_28)
    return im_operator

def get_ordered_contour(im):
    '''
    Contour detection function based on the simple images we have.
    
    Input: grayscale image
    Output: contour of the main feature ( In our application, handwritten digits )
    
    '''
    #im = low_pass_filter(im,3)
    contours = measure.find_contours(im, 50)
    contours_array = []
    for n, contour in enumerate(contours):
        for x,y in contour:
            contours_array.append((x,y))
    contours_array = np.array(contours_array).astype(int)
        
    return contours_array

def get_pixel(im):
    #im_filtered = low_pass_filter(im,3)
    coordinate = []
    
    row,col = im.shape
    for r in range(row):
        for c in range(col):
            coordinate.append((r,c))
    return coordinate

def is_inside_contour(contour, non_zero_pixels):
    counter = 0
    for coord in non_zero_pixels:
        dist = cv2.pointPolygonTest(contour,coord,True)
        if dist > 0:
            counter += 1
    return counter

def compute_perimeter(contour_list):
    coord_tmp = []
    counter = 0
    nearby_point = 0
    for coord in contour_list:
        coord_tmp.append(coord)
        if counter != 0:
            if (coord_tmp[0][0] == coord_tmp[1][0]) or (coord_tmp[0][1] == coord_tmp[1][1]):
                nearby_point += 1
            coord_tmp.pop(0)
        counter += 1
    p_2 = (len(contour_list) - nearby_point)*math.sqrt(2) + nearby_point - 1
    return p_2

def compacity(im_inv) :
    im = (np.max(im_inv)-im_inv)
    contour = get_ordered_contour(im) 
    region_coord = get_pixel(im)
    contour_list = contour.tolist()
    
    # Computing Area A_2
    i = is_inside_contour(contour, region_coord)
    b = len(contour)
    area = (b/2 + i-1)
    
    # Computing perimeter P_2
    perimeter = compute_perimeter(contour_list)
    
    # Computing compacity
    compacity = perimeter**2/area
    return compacity

def init_operator_id() : 
    operator_typical = []
    op_symbol = ['+', '=', '-', '/', '*']
    for i in range(len(im_operator)):
        operator_typical.append(compacity(im_operator[i]))
    return operator_typical, op_symbol

def init_operator_id() : 
    operator_typical = []
    op_symbol = ['+', '=', '-', '/', '*']
    im_operator = initiate_symbol_detection()
    for i in range(len(im_operator)):
        operator_typical.append(compacity(im_operator[i]))
    return operator_typical, op_symbol  

def find_operator_id(im_input):
    im_filt= cv2.cvtColor(im_input,cv2.COLOR_BGR2GRAY)
    contours = measure.find_contours(im_filt, 90)
    if len(contours) == 2:
        return '='
    elif len(contours) == 3:
        return '/'
    else:
        im_comp = binaryImage(im_filt)
        comp_test = compacity(im_filt)
        dist_ref = 1000
        for i in range(len(operator_typical)):
            dist = abs(comp_test-operator_typical[i])
            if dist < dist_ref:
                dist_ref = dist
                id_op = i
    return op_symbol[id_op]           


import argparse

parser=argparse.ArgumentParser(description='Read the video and output the equation')
parser.add_argument('--input', nargs=1, required=True, type=str)
parser.add_argument('--output', nargs=1, required=True, type=str)
args=parser.parse_args()

video_path = args.input[0]

video_frames = read_video(video_path)
path = compute_path(video_frames)
first_frame = symbol_detect(video_frames[0])
symbols_masks = shape_extraction(first_frame)
symbols_images,symbol_full, symbol_cm = create_object_image(symbols_masks,video_frames[0],5)
symbols_bin_im = features_extraction(symbols_images)
model = keras.models.load_model('LeNet_model')

operator_typical,op_symbol = init_operator_id()

symbols_images, symbols_full, symbol_cm = create_object_image(symbols_masks,video_frames[0],10)
w,h,c = video_frames[0].shape
dist_between_symbols = []

output = video_frames.copy()
currently_detecting = False
intersection = False
counter = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (40,400)
fontScale              = 1
fontColor              = (0,150,0)
lineType               = 2

text = ''
for n,arrow_coords in enumerate(path):
    dist = []
    for coord in symbol_cm:
        norm2_dist = cv2.norm(np.array(coord),np.array(arrow_coords),cv2.NORM_L2)
        dist.append(norm2_dist)
    k = np.argmin(np.array(dist))
    
    if close(arrow_coords,symbol_cm[k]):
        intersection = True
    else:
        currently_detecting = False
        intersection = False
        
    if intersection and not currently_detecting:
        currently_detecting = True 
        if not counter%2:
            digit = symbols_bin_im[k].astype(float)
            label = int(model.predict_classes(digit.reshape(1,28,28,1)))
            print('digit:', label)
            text= text + str(label) + ' '
        else:
            symbol = find_operator_id(symbols_full[k])
            print('symbol:',symbol)
            text= text + symbol + ' '
            if symbol == '=':
                text = text + str(eval(text[:-2]))
                print(text)
        counter += 1
    output[n] = cv2.putText(output[n],text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    output[n] = draw_path(output[n],path,w,h,n)

output_path = os.path.join(args.output[0], 'Output.mp4')

writer = imageio.get_writer(output_path, fps=2)
for im in output:
    writer.append_data(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
writer.close()