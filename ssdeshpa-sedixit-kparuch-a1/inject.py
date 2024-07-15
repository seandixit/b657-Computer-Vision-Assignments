import numpy as np
from PIL import Image, ImageDraw
import sys

def shuffle_dict(d, assortment):
    """
    Shuffle the dictionary based on the assortment
    """
    shuffled_dict = d.copy()
    for i in range(len(assortment)):
        shuffled_dict.update({assortment[i]+1 : d[i+1]})   
    return shuffled_dict


def file_to_dict(filename):
    """
    Converts answers file into a dictionary with keys as the question number and values as the answer
    """
    result = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                result[int(parts[0])] = parts[1]
    return result

def encode_text(answer_path, form_image_path, output_image_path, assortment):
    """
    Create barcode onto form_image and save onto injected.jpg
    """
    print("Encoding text...")
    # Map answers to some width
    answer_to_height_mapping = {
    "A": 6, "B": 7, "C": 8, "D": 9, "E": 10,  # Single selections
    "AB": 11, "AC": 12, "AD": 13, "AE": 14, "BC": 15, "BD": 16, "BE": 17, "CD": 18, "CE": 19, "DE": 20,  # Two selections
    "ABC": 21, "ABD": 22, "ABE": 23, "ACD": 24, "ACE": 25, "ADE": 26, "BCD": 27, "BCE": 28, "BDE": 29, "CDE": 30,  # Three selections
    "ABCD": 31, "ABCE": 32, "ABDE": 33, "ACDE": 34, "BCDE": 35,  # Four selections
    "ABCDE": 36  # All selections
}

    form_image = Image.open(form_image_path)
    img_width, img_height = form_image.size

    # Create a PIL ImageDraw object
    draw = ImageDraw.Draw(form_image)

    q_and_ans = file_to_dict(answer_path)
    q_and_ans = shuffle_dict(q_and_ans, assortment) 

    x = 0
    y = 15
    draw.rectangle([x, y, x + 20, y + 50], fill="black", outline="black", width=1)
    y += 57
    # Draw rectangles based on the number of questions
    for question_number, answer in q_and_ans.items():
        if (y > img_height - 90):
            x=20
            y=15
        height = answer_to_height_mapping.get(answer)
        # Draw rectangle starting from (x, y) with width 100 pixels and height 20 pixels
        draw.rectangle([x, y, x + 20, y + height], fill="black", outline="black", width=1)
        y += height 
        # Move to the next position
        y += 7
    
    draw.rectangle([x, y, 20, y + 50], fill="black", outline="black", width=1)

    # Save the image
    form_image.save(output_image_path)

    return "Encoding injected!"
        
# ---------------------------------------------------------
if __name__ == "__main__":
    input_image_path = sys.argv[1]
    answers_file_path = sys.argv[2]
    output_image_path = sys.argv[3]
    assortment = [36, 81, 11, 21, 43, 26, 9, 14, 59, 74, 69, 78, 19, 46, 25, 75, 83, 4, 71, 50, 
              31, 38, 45, 66, 13, 57, 32, 37, 77, 33, 47, 30, 28, 48, 63, 23, 55, 0, 72, 24,
              79, 8, 70, 20, 18, 51, 17, 80, 7, 60, 12, 73, 84, 1, 64, 53, 49, 58, 56, 34, 6, 
              22, 10, 15, 27, 76, 65, 68, 52, 35, 2, 3, 5, 61, 29, 82, 54, 67, 42, 40, 41, 62, 
              44, 16, 39] # key=1 goes to 36, key=2 goes to 81, etc.
    # ^ allows us to randomize shuffle the answers and then unshuffle when extracting
    print(encode_text(answers_file_path, input_image_path, output_image_path, assortment))