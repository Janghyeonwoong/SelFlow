import numpy as np
import cv2 as cv
import os

one_step = 45
half_step = one_step / 2

def read_flo(file):
    with open(file, 'rb') as f:
        magic, = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w, h = np.fromfile(f, np.int32, count=2)
            data = np.fromfile(f, np.float32, count=2*w*h)
            data2D = np.resize(data, (w, h, 2))
            return data2D

def rad_to_deg(rad):
    return rad * 180 / np.pi

vectorize_function1 = np.vectorize(rad_to_deg)

def angle_to_direction(pixel_value):
    if pixel_value < -180 + half_step:
        return 5
    elif pixel_value < -180 + one_step + half_step:
        return 4
    elif pixel_value < -180 + one_step * 2 + half_step:
        return 7
    elif pixel_value < -180 + one_step * 3 + half_step:
        return 2
    elif pixel_value < -180 + one_step * 4 + half_step:
        return 1
    elif pixel_value < -180 + one_step * 5 + half_step:
        return 8
    elif pixel_value < -180 + one_step * 6 + half_step:
        return 3
    elif pixel_value < -180 + one_step * 7 + half_step:
        return 6
    else:
        return 5  

vectorize_function2 = np.vectorize(angle_to_direction)   

def set_direction(image_patch, percentage=0.8):
    area = image_patch.shape[0] * image_patch.shape[1] 
    pixels_angle = np.arctan2(image_patch[:,:,1], image_patch[:,:,0])
    pixels_angle = pixels_angle.flatten()
    pixels_angle = vectorize_function1(pixels_angle)
    pixels_angle = vectorize_function2(pixels_angle)
    histogram = make_histogram(pixels_angle, bins = [1,2,3,4,5,6,7,8])
    if np.max(histogram) > area * percentage:
        return np.argmax(histogram) + 1
    else:
        return 10

def make_histogram(values, bins):
    histogram, _ = np.histogram(values, bins = bins)
    return histogram 

def draw_line(image, direction):
    center = (int(image.shape[0] / 2), int(image.shape[1] / 2))
    
    center_y = center[0]
    center_x = center[1]

    right_x = image.shape[0] - 10
    left_x  = 10
    up_y    = 10
    down_y  = image.shape[1] - 10

    if direction == 1:
        cv.arrowedLine(image, center, (right_x, center_y), (0,0,0), thickness=2)
    elif direction == 2:
        cv.arrowedLine(image, center, (right_x, up_y), (0,0,0), thickness=2)
    elif direction == 3:
        cv.arrowedLine(image, center, (center_x, up_y), (0,0,0), thickness=2)
    elif direction == 4:
        cv.arrowedLine(image, center, (left_x, up_y), (0,0,0), thickness=2)
    elif direction == 5:
        cv.arrowedLine(image, center, (left_x, center_y), (0,0,0), thickness=2)
    elif direction == 6:
        cv.arrowedLine(image, center, (left_x, down_y), (0,0,0), thickness=2)
    elif direction == 7:
        cv.arrowedLine(image, center, (center_x, down_y), (0,0,0), thickness=2)
    elif direction == 8:
        cv.arrowedLine(image, center, (right_x, down_y), (0,0,0), thickness=2)
    else:
        cv.circle(image, center, int(center_x/2), (0,0,0), thickness=2)

flo_dir = 'F:/result/1'
result_dir = 'F:/vector'

height = 576
width  = 576

patch_height = int(height / 8)
patch_width  = int(width / 8)

mask = np.ones([576,576,3]) * 255
mask[0:72,:,:] = 0
mask[72:144,:144,:] = 0
mask[72:144,-144:,:] = 0
mask[:,:72,:] = 0
mask[:,-72:,:] = 0
mask[-72:,:,:] = 0
mask[-144:-72,:144,:] = 0
mask[-144:-72,-144:,:] = 0


files = os.listdir(flo_dir)
flo_files = []

for path in files:
    if 'flo' in path:
        flo_files.append(path)

flo_files.sort(key = lambda x : int(x.split('.')[0]))
flo_files = [os.path.join(flo_dir, path) for path in flo_files]

i = 0
length = len(flo_files)

for flo in flo_files:

    directions = np.zeros([8,8])
    direction_image = np.ones([height,width,3], dtype=np.uint8) * 255
    flo_mat = read_flo(flo)
    for y in range(8):
        for x in range(8):
            patch = flo_mat[y*patch_height:(y+1)*patch_height, x*patch_width:(x+1)*patch_width]
            direction = set_direction(patch)
            directions[y][x] = direction
            draw_line(direction_image[y*patch_height:(y+1)*patch_height, x*patch_width:(x+1)*patch_width], direction)

    result = direction_image * mask
    file_name = flo.split('/')[-1]
    file_name = file_name.split('.')[0]
    cv.imwrite(os.path.join(result_dir, file_name + '.jpg'), result)
    i = i + 1
    print('%d/%d' %(i, length))