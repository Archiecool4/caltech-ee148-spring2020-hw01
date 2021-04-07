import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    red = I[:, :, 0]
    green = I[:, :, 1]
    blue = I[:, :, 2]

    thresh = 200

    # Take dominant red color
    lights = np.asarray([red[x, y] > thresh and green[x, y] < thresh and 
        blue[x, y] < thresh for x in range(len(red)) for y in 
        range(len(red[0]))]).reshape(red.shape)

    # Plot image 
    # plt.subplot(121)
    # plt.imshow(I)
    # plt.title('Original Image')
    # plt.subplot(122)

    # Circle Hough Transform
    acc = np.zeros((red.shape[0] + 1, red.shape[1] + 1, 20))
    for r in range(1, 3):
        print(r)
        idxs = np.where(lights == 1)
        for x, y in zip(idxs[0], idxs[1]):
            for theta in range(360):
                b = int(y - r * np.sin(theta * np.pi / 180))
                a = int(x - r * np.cos(theta * np.pi / 180))
                if a >= acc.shape[0] or b >= acc.shape[1]:
                    continue
                acc[a, b, r] += 1

    # Bounding boxes
    circle_thresh = 100
    # plt.imshow(I)
    # plt.title('Labeled Image')
    idxs = np.where(acc > circle_thresh)
    for x, y, r in zip(idxs[0], idxs[1], idxs[2]):
        # plt.gca().add_patch(mpatches.Rectangle((y-r, x-r), r, r, color='green'))
        bounding_boxes.append([int(x-r), int(y-r), int(x), int(y)])

    plt.show()

    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = './data/RedLights2011_Medium/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# Bad:  RL-011, RL-167, RL-245
# Good: RL-020, RL-182, RL-264, 
# file_names = ['RL-020.jpg', 'RL-182.jpg', 'RL-264.jpg']

preds = {}
for i in range(len(file_names)):
    print('{:.0f}%'.format(100 * (i / len(file_names))), end='\r')
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
