import numpy as np
from PIL import Image, ImageFilter, ImageOps
import sys

def unshuffle_dict(shuffled_dict, assortment):
    """
    Unshuffle the dictionary based on the assortment
    """
    unshuffled_dict = {}
    for index, key in enumerate(assortment):
        try:
            unshuffled_dict[index+1] = shuffled_dict[key+1]
        except KeyError: 
            pass
    return unshuffled_dict


def decode_image(injected_image_path, write_into_path, assortment):
    """
    Decode the injected image to retrieve the answers.
    written into write_into_path
    """
    print("Decoding image...")
    # the key to decoding is the map
    answer_to_height_mapping = {
    "A": 6, "B": 7, "C": 8, "D": 9, "E": 10,  # Single selections
    "AB": 11, "AC": 12, "AD": 13, "AE": 14, "BC": 15, "BD": 16, "BE": 17, "CD": 18, "CE": 19, "DE": 20,  # Two selections
    "ABC": 21, "ABD": 22, "ABE": 23, "ACD": 24, "ACE": 25, "ADE": 26, "BCD": 27, "BCE": 28, "BDE": 29, "CDE": 30,  # Three selections
    "ABCD": 31, "ABCE": 32, "ABDE": 33, "ACDE": 34, "BCDE": 35,  # Four selections
    "ABCDE": 36  # All selections
}

    # Open the injected image and convert to grayscale
    injected_image = Image.open(injected_image_path).convert("L")

    # change size of image to fixed size 
    injected_image = injected_image.resize((1700, 2200))
    img_width, img_height = injected_image.size
    
    # threshold (added after realizing noise affects)
    threshold = 80
    injected_image = injected_image.point(lambda p: p > threshold and 255)

    # min and max filter
    kernel_size = 3
    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    injected_image = injected_image.filter(ImageFilter.MinFilter(kernel_size))
    injected_image = injected_image.filter(ImageFilter.MaxFilter(kernel_size))
    #injected_image.show()

    # Threshold for considering an average value as black 
    black_threshold = 20  
    
    # Threshold for considering a height match
    height_threshold = 1

    # Initial position
    x = 0
    y = 0

    # Flag to indicate whether we are currently reading a black bar
    reading_black = False

    # Flag to indicate whether we have encountered the start of the barcode
    start_encountered = False

    # List to store the decoded answers
    decoded_answers = {}

    roi_height = 18

    height_count=0
    while x < img_width:
        # Get pixel value
        #pixel_value = injected_image.getpixel((x, y))
        sum_pixel_values=0
        for dx in range(roi_height):
            pixel_value = injected_image.getpixel((x+dx, y))
            sum_pixel_values += pixel_value

        average_pixel_value = sum_pixel_values / roi_height
        
        # Check if the pixel is black
        if average_pixel_value < black_threshold:
            # If we were not previously reading a black bar, start reading
            if not reading_black:
                reading_black = True
                #print((x,y))
                height_count = 1
            else:
                height_count += 1
                if height_count >= 46: # 50 is height of start bar
                    if start_encountered:
                        break  # Stop decoding when encountering a black bar of 80 pixels
                    else:
                        start_encountered = True
                        height_count = 0 # Reset height count 
        else: 
            # If we were previously reading a black bar, check for the height
            if reading_black:
                # Check if the height matches any answer
                for answer, height in answer_to_height_mapping.items():
                    if abs(height_count - height) <= height_threshold:
                        decoded_answers.update({len(decoded_answers)+1: answer})
                        break
                reading_black = False
        
        # Move to the next pixel
        y += 1
        if y >= img_height - 50:
            if (start_encountered):
                y = 0
                x += 21
            else:       # if, say, we have white padding to the left of barcode 
                y = 0
                x += 2

    write_decoded_answers(decoded_answers, write_into_path, assortment)
    return "Decoded answers saved!"

def write_decoded_answers(decoded_answers, output_file, assortment):
    """
    Write the decoded answers to an output text file.
    """
    # First one doesn't matter (part of starting bar)
    decoded_answers.pop(1, None)
    # Shift all keys to the left by 1
    decoded_answers = {key - 1: value for key, value in decoded_answers.items()}
    
    unshuffled = unshuffle_dict(decoded_answers, assortment)
    with open(output_file, 'w') as file:
        for question_number, answer in unshuffled.items():
            file.write(f"{question_number} {answer}\n")


def count_same_lines(file1, file2):
    """
    Count the number of lines that are the same between two text files.
    """
    count = 0
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            if line1.strip() == line2.strip():
                count += 1
    return count

# ---------------------------------------------------------
if __name__ == "__main__":
    input_image_path = sys.argv[1]
    answers_file_path = sys.argv[2]
    assortment = [36, 81, 11, 21, 43, 26, 9, 14, 59, 74, 69, 78, 19, 46, 25, 75, 83, 4, 71, 50, 
              31, 38, 45, 66, 13, 57, 32, 37, 77, 33, 47, 30, 28, 48, 63, 23, 55, 0, 72, 24,
              79, 8, 70, 20, 18, 51, 17, 80, 7, 60, 12, 73, 84, 1, 64, 53, 49, 58, 56, 34, 6, 
              22, 10, 15, 27, 76, 65, 68, 52, 35, 2, 3, 5, 61, 29, 82, 54, 67, 42, 40, 41, 62, 
              44, 16, 39] # key=1 goes to 36, key=2 goes to 81, etc.
    # ^ allows us to randomize shuffle the answers and then unshuffle when extracting
    decoded_answers = decode_image(input_image_path, answers_file_path, assortment)
    print(decoded_answers) # prints "...saved!"

    #same_lines_count = count_same_lines("test-images/a-3_groundtruth.txt", answers_file_path)
    #print("Accuracy=", same_lines_count / len(open("test-images/a-3_groundtruth.txt").readlines()))