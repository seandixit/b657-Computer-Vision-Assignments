from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import sys
import os

def check_bubble(image, x1, y1, x2, y2,a):
    #cropping the input img in bounding box
    img_crop = image.crop((x1, y1, x2, y2))
    #grayscaling
    gray_cropped = img_crop.convert('L')
    #  creating histograms 

    histo = gray_cropped.histogram()

    # calculating the total number of pixels 
    tp = sum(histo)

    # treshholding on the histogram
    threshold = sum(i*histo[i] for i in range(256)) / tp

    # checking for the treshold 
    return threshold < a

def left_qn_answer(path):
    # Load the image
    input_image = Image.open(path)

    # Calculated the coordinates manually and conditioned based on this
    x1 = 150
    y1 = 680
    w = 60
    h = 40  
    y1toy2 = 47.5  # vertical offset in between qns

    # total number of qns
    total_q = 85
    x = []
    # Loopinbg over each qn
    for qn in range(total_q):

        # coordinates based on the current option 
        px1 = x1 + (qn // 29) * 430
        px2 = px1 + w
        py1 = y1 + (qn % 29) * y1toy2
        py2 = py1 + h
        a = 250
        # Checking for marked option
        ynbubble = check_bubble(input_image, px1, py1, px2, py2,a)
        # Appending the qn number and marked choice 
        x.append((qn + 1, 'x' )if ynbubble else (qn + 1,""))
    # print(x)
    return x

# def green_rectangle(px1, py1, px2, py2):
  

def bubbled_ans(path):
    #loading the img
    input_image = Image.open(path)

    #the initial coord for q1 are
    x1 = 270
    y1 = 680
    w = 30 #width
    h = 47 #1a to 2a #height
    y1toy2 = 47

    #for each qn there are 5 options
    no_of_options = 5

    #total no of qns in the omr
    total_q = 85
    y = []
    #checking each and every qn
    for qn in range(total_q):
        bubbled = []
        #in every qn iterating through each option to check if it is bubbled
        for opt_no in range(no_of_options):
            #getting the coordinates for each option
            px1 = x1 + opt_no * 60 + (qn // 29) * 430
            px2 = px1 + w
            py1 = y1 + (qn % 29) * y1toy2
            py2 = py1 + h
            a = 195
            #checking if the current option is bubbled
            ynbubble = check_bubble(input_image, px1, py1, px2, py2,a)
            #appending the option if it is bubbled
            if ynbubble:
                bubbled.append(chr(65 + opt_no))
                # green_rectangle(px1, py1, px2, py2)
        y.append((qn + 1, ' '.join(bubbled)))
        # print(f"{qn + 1} {' '.join(bubbled)}")
    return y

#####################################################################################
# Outlining the highlighted answers in the image

def is_area_filled(image, corners, threshold_black=250):
    image = image.convert('L')  
    corners_array = np.array(corners)
    x_min = corners_array[:, 0].min()
    x_max = corners_array[:, 0].max()
    y_min = corners_array[:, 1].min()
    y_max = corners_array[:, 1].max()

    # so we kept a counter to calculate the number of black pixels present in the area of interest which are the squares
    black_pixels = 0

    for y in range(int(y_min), int(y_max)):
        for x in range(int(x_min), int(x_max)):
            pixel_value = image.getpixel((x, y))
            if pixel_value == 0:  # Black pixel value is 0
                black_pixels += 1

    # we will now be checking if the number of black pixels exceeds the threshold so that if they do then we can add in that 
    # condition later
    if black_pixels > threshold_black:
        return True
    else:
        return False
    
##############################################################################################################################
    
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 pichu_devil.py input.jpg")
    
    print("Recognizing form.jpg")

    input_path = sys.argv[1]  
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(script_dir, "test-images", input_path)

    #the options that are bubbled in each question are extracted
    p=bubbled_ans(image_path)
    #marking x if any letter is written to the left of the question number
    q=left_qn_answer(image_path)

    #adding the whole information collected into the text file
    with open('output.txt', 'w') as file:
    #writing into the file
      for i, j in zip(p, q):
          k1, v1 = i
          k2, v2 = j
          file.write(f"{k1} {v1} {v2}\n")

    original_image = Image.open(image_path)
    gray_image = original_image.convert("L")
    gray_array = np.array(gray_image)

    # we will be thresholding the image to create a binary image
    threshold = 100  
    binary_image = gray_array < threshold
    contours = measure.find_contours(binary_image, 0.8)

    squares_rectangles = []
    top_left_corners = []
    for contour in contours:
        x_min, y_min = contour.min(axis=0)
        x_max, y_max = contour.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = width / height
        if 0.8 <= aspect_ratio <= 1.2 and width > 30 and height > 30: 
            # Adjusting coordinates to draw outline outside the box
            squares_rectangles.append((y_min - 5, x_min - 5, y_max + 5, x_max + 5))  
            top_left_corners.append([(y_min, x_min), (y_max, x_min), (y_max, x_max), (y_min, x_max)])  

    draw = ImageDraw.Draw(original_image)

    for corners, rectangle in zip(top_left_corners, squares_rectangles):
        if is_area_filled(gray_image, corners):
            draw.rectangle(rectangle, outline="green", width=5)

    output_path = "scored.jpg"
    original_image.save(output_path)
    original_image.show()

    print("Result saved as:", output_path)

