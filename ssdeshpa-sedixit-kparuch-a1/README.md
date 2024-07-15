# ssdeshpa-sedixit-kparuch-a1
# How to Run
We use the following libraries:
- numpy
- PIL
- sys
- matplotlib.pyplot
- skimage
- os

**Example Usage**

To run grade.py:
python3 grade.py form.jpg output.txt

To run inject.py:
python3 inject.py test-images/a-3.jpg test-images/a-3_groundtruth.txt injected.jpg

To run extract.py:
python3 extract.py injected.jpg output_test.txt

# Report
## Grading
### Design and Philosophy
The idea is to extract the answers bubbled in the OMR sheet and to store it in the text file named output.txt. Additionally if any letter was written to the left of the question number ‘x’ is marked after the option number in the output.txt. The main functions of the code are bubbled_ans() and left_an_answer(). The function bubbled_ans() detects the bubbled answers for each of the question based on the predefined coordinates and using a threshold value that is derived from the histogram. Other function left_an_answer() identifies the questions where the options are written to the left of the question number. Both of these functions used image manipulation and analysis, by using the pillow library to extract efficiently. So the results obtained are stored as a tuple containing the question number, the options and ‘x’ if the letter is written to the left of the question number. This is added to the output.txt file. 

The code has clarity, modularity and flexibility that makes easier to detect the option from the OMR sheet. The code is segmented in different functionalities such as bubble detection and left side written option detection. Furthermore, the code supports variation in OMR sheet, according to the scanned image the threshold values and the coordinates can be changed. 

### Base Implementation

The code uses Pillow library to detect the options that are bubbled in the OMR sheet. check_bubble() is a function that is used to check if the option of a question is bubbled or not. It determines whether it is marked or not based on the threshold value. The bubbled_ans() function analysis each question by going through each and every option present in the question on the OMR sheet by calculating coordinates of each option and then checks if it is marked. The results are stored in a list it contains the question number, marked options and then ‘x’ if letter is present to the left. 

### Improvements
Finding the initial coordinates is challenging. Initially the width and height of the coordinates is not known clearly resulting in the creation of the bounding boxes in between the question. 
![image](https://media.github.iu.edu/user/24344/files/4c9ee4b8-506b-494d-9669-b92300454253)
This challenge was overcome by trying to create different boundary boxes and tracing out the pattern between them to find the width and height of the option.
![image](https://media.github.iu.edu/user/24344/files/4637ce18-3ed5-487b-aa59-754b9f5755cb)
Additionally finding the threshold value is challenging. If the taken threshold value does not meet the requirement, the marked options were not detected. And also the letter written to the left of the question number is extracted with certain threshold. 

![image](https://media.github.iu.edu/user/24344/files/8e4581f9-6048-41d0-b8f3-1d3a8ab84abb)

This challenge has been overcome by finding the average of pixels values of the boundary box and then fixated the threshold value.

![image](https://media.github.iu.edu/user/24344/files/46d86095-7574-4d89-ac93-76b2443cba40)

### Limitations and Future

The code’s accuracy depends on the quality of the scanned images. It is sensitive to the variations in noise levels and distortions. The analysis of the histogram and the threshold value reduces the efficiency of the code particular when dealing with the OMR sheets.

So, to address these limitations the code can be integrated with machine learning algorithms like CNN that can give better accuracy, can also detect the marked options when the image is affected with noise.  Several image processing techniques can be used to enhance the quality of the scanned images, so that the code can deal with the variations that are present in the image.

## Drawing and Highlighting Answers (Filled Box) in the image 
### Design Philosophy 
The idea guiding the approach was to directly manipulate the positional information encoded in the image coordinates. By working on the coordinates themselves, the idea was to write an algorithm which could discern meaningful patterns like squares/rectangles within the image for analysis and processing. This concept underpins the whole process, in which the interpretation and manipulation of the spatial connections in the image is the central focus of each phase.

Drawing both vertical and horizontal lines was how we started experimenting. This first step was to acquire an understanding of where the image's first row of boxes should be placed. We were able to comprehend the vertical and horizontal axis alignment of the boxes as well as their spatial relationships between them better by viewing these lines. We were able to improve the algorithm's comprehension of the image's architecture and identify regions of interest thanks to the experimental phase, which gave us vital context about the coordinates. 


To facilitate more efficient processing, we cropped the image to concentrate on the lower portion. Next, we manually determined the starting coordinates of a rectangle inside the reduced area. The coordinates were expressed as a list of tuples, where each tuple indicated the corner of the rectangle's (x, y) location. We started an iterative approach to gain a deeper understanding of the region's layout. We shifted the rectangle horizontally and verified that every spot was occupied on each cycle. In order to determine if the region matching to the original corner coordinates was filled, We first printed those coordinates. If it was full, we sketched the square using PIL library's ImageDraw function. We then estimated the necessary shift for each horizontal location going forward using a fixed horizontal shift assumption of 60 pixels. We repeated the process of checking for full regions and creating rectangles, updating the x-coordinates accordingly. We printed the coordinates of every rectangle for reference during this procedure. After finishing a horizontal iteration, We also determined the shift for the subsequent vertical position and updated the rectangle's y-coordinates appropriately. We were able to methodically investigate and comprehend the geographical distribution of the region of interest inside the image thanks to this iterative process.

<img width="302" alt="image" src="https://media.github.iu.edu/user/21691/files/a38848a8-d215-494d-86cf-a4d70e4aec1e">


<img width="304" alt="image" src="https://media.github.iu.edu/user/21691/files/48efa292-0171-425a-8388-1c204fc24b71">


Since the starting coordinates were manually determined, they were specific to the particular image being analyzed. Consequently, these coordinates could not appropriately align with the structures in other images and could not capture the regions of interest when applied to other images. This inflexibility made it necessary to create a more reliable and automated method for locating and analyzing regions of interest in images, which guaranteed accurate and consistent outcomes across the other images.

In order to determing the corner coordinates dynamically so as to move the coordinates, we created a script that would identify the upper left square and its left top corner. First, our script examined the picture to find possible rectangles and squares. The image was first converted to grayscale, a binary threshold was applied, then the scikit-image package was used to locate contours in the image that represented different forms. Then, using predetermined parameters like aspect ratio, width, and height, we eliminated squares and rectangles. We determined each top-left corner's Euclidean distance to a reference point after determining the candidate squares and rectangles (240, 10). Either randomly or in light of the aforementioned experimental findings, this reference point was selected. The corner that was closest to the top-left was the one that measured the least distance to the reference point. We created a green dot around the closest top-left corner on a blank image that was the same size as the original image in order to visually highlight it. The corner that was chosen as the beginning point for moving the square was made easier to detect by this green dot. Once the closest top-left corner was identified, we attempted to create a new set of coordinates for a smaller square by modifying the coordinates in relation to the closest corner. The specifications of the application were used to calculate the new square's size. The process of locating and moving the square from the determined beginning point was finally completed by the script, which drew the new square with modified coordinates on the original picture. Targeted changes and analysis were made possible by this technology, which allowed for exact alteration of the square's position depending on spatial connections within the image.

<img width="653" alt="image" src="https://media.github.iu.edu/user/21691/files/aa5300bc-2416-4bf6-9e14-2abc24798926">

### Base Implementation

The primary approach used in the base implementation was to automatically outline the responses that were highlighted in an input image. The script first loaded and grayscaled the picture that was sent in as a command-line parameter. It then used a binary threshold to produce a binary picture, where pixels with values above and below the threshold were designated as white and black, respectively. The scikit-image package was then used to identify contours in the binary picture that represented different forms. Squares and rectangles were chosen from these contours by filtering them according to predetermined standards including aspect ratio, width, and height. The script first recognized the important shapes and then created outlines around them on the original image, focusing on areas that were probably going to have highlighted responses. 

A threshold of black pixels inside the area of interest was considered manually by experimentation to assess if a region was filled. The contour was created in thick gray if the quantity of black pixels was more than this threshold, signifying a filled region. Eventually, the finished product—an picture with highlighted areas—was stored and shown. This procedure made it possible to automatically recognize and highlight parts in the input image that could be important, which made it easier to undertake additional processing or analysis.


## Injecting and Extracting
### Design Philosophy
Initially, our idea was to encode the entire answers txt file into an image that would hold the text as a secret message, which would then be injected into the blank form. We stumbled across a video by Tech Raj that outlined a steganography technique that allows us to encode secret messages into images (<https://www.youtube.com/watch?v=_KX8ORUA_98&t=288s>). The technique involved taking the ascii 8-bit representation of each letter in our answers.txt file, and encoding it into 3 pixels in the image (9 RGB intensity values altered) by setting the corresponding RGB value to be even if the bit is 0, or else odd. The 9th RGB color intensity value is used to indicate termination. We created a naive implementation of the technique discussed in the video, which was heavily inspired by the code shwon in the video. However, after injecting the encoded image onto the form, while trying to decode the text from the encoded image in the form in extract.py, we saw that the image had different color intensity values, making it impossible to decode.
![image](https://media.github.iu.edu/user/18309/files/e4bc496b-3091-416c-9c65-6169f0aa5e09)

Notice how the image with answers encoded is brighter than the snippet we extracted from injected.jpg in extract.py. We tried linearly changing the brightness and checking if we can then decode the image, but to no avail. The change in color intensity values may be due to image compression, or some underlying changes made while injecting the image onto the form or extracting the image from the form. We also realized: to make our technique robust to color intensity changes, which is inevitable when rescanning, our technique must not be sensitive to color intensity changes. This technique relied on the individual color intensities being even or odd, and so any minor change in color intensity when rescanning (in a practical scenario) would make it impossible to decode. 

After some brainstorming, we thought of creating a barcode system to encode the answers. Our idea was simple: have 85 black bars on top of the form (one for each question), with each bar width indicating the answer it represents. We created a mapping that mapped each possible answer (so, 31 string values) to widths. This acted as a sort of passkey that we use to encode and to decode. 

![image](https://media.github.iu.edu/user/18309/files/5abb8d73-48a9-4077-8854-7f9dc862dc96)
*Mapping each possible answer to a bar width*

We made the black bars be fairly tall so, if while rescanning a few pixels from the top of the form are left out, we can still decode the bars. Furthermore, it being on top of the form meant there would be a lower chance of student’s accidentally writing over the barcode (but, as you will see later, this is something we were able to mitigate). So, when encoding, we draw black bars on top of the form using the answer to width mapping, with start and end black bars being wider than any possible answer width. Then, when decoding, we look for the start bar, and after finding it, we refer to the answer to width mapping to figure out the answers from the width of the bars until we reach the end bar.

### Base Implementation
Our method to test accuracy was to simply divide how many lines are different between answers.txt (from inject.py) and output.txt (from extract.py) and dividing it by 85.
Our base implementation worked fairly well. It:
- Was robust against slight changes in size: by having 3 pixel difference between every width value in our mapping (e.g “A” mapped as width=6 and “B” mapped as width=9), and approximating the size of bars when decoding, allowing for +-1 pixels when comparing bar width to mapping (e.g pixel width=7 mapped as “A”). Furthermore, if while rescanning some of the top pixels are cut off, we are still able to decode.
- Was robust against change in color: we convert the image into grayscale and set pixels with value less than a threshold to be black before decoding.
- Was robust against student writing near barcode: because we only start counting black pixels when the starting bar is encountered.
![image](https://media.github.iu.edu/user/18309/files/02a77cb9-3d07-4c49-9556-d0a905893efe)
- Was robust against noise: We generated some gaussian noise on the image and saw that we were able to decode with 100% accuracy up till noise with standard deviation of 50.
![image](https://media.github.iu.edu/user/18309/files/c0a37ff6-0239-4a62-93f2-3eb94be8f8af)

However, there were some improvements to be made.

### Improvements 
Firstly, because each bar width represents an answer, we realized that a student can know if two question answers are the same if they assume, as one would instinctively, that the bars represent answers in chronological order. So, we introduce a way to shuffle the order of answers in inject.py, and then unshuffling them into the original, chronological order in extract.py. In inject.py, the answers are stored in a dict with keys as question numbers and value as the answer. We shuffle this dictionary using a list that maps each index of the dict to a different index. This list, we call assortment, also acts as a sort of passkey (so a student can only decipher the code if they somehow remember the assortment as well as the answer to width mapping). The assortment list is then used in extract.py to unshuffle, or put the dict key/values to their original index. 

![image](https://media.github.iu.edu/user/18309/files/9cf9049f-fe63-4281-988a-9df39c1c4696)

We made a slight oversight in our technique: we forgot to test for whether the page can fit the barcode. We tested this by creating an answers.txt that had “ABCDE” as the answer for every question (which had the longest width in our mapping) and we saw that the barcode extended way past the width of the form.
![image](https://media.github.iu.edu/user/18309/files/8254fdaf-3d5e-43b2-88ef-114d18ca4930)

To fix this, we did the following, in order:
- Decreased the difference between each consecutive answer width mapping from 3 to 1, and shifted the start block to the left. 

![image](https://media.github.iu.edu/user/18309/files/90543c0e-f3c8-4ad1-be0a-d4e2d5bf2448)
- Upon realizing there was still significant overflow, we made the barcode vertical on the left edge of the form instead.

![image](https://media.github.iu.edu/user/18309/files/bd825619-bb27-41e4-a888-b64d0993677c)
- It was still overflowing. We then had to modify our approach to have a second column of bars if we detect overflowing (i.e y is close to the form's height). When decoding, when y is close to the form’s height, we increment x by 20.

![image](https://media.github.iu.edu/user/18309/files/a7e83be4-bc9c-4737-8736-c577b5b19128)

When we downsized the injected.jpg while testing, we saw colors bleeding between the bars which can make the decoding algorithm think two bars are conjoined, which can lead it to stop decoding thinking it reached the end bar. So, we made the gap between the bars higher.

![image](https://media.github.iu.edu/user/18309/files/22fdd581-286a-49da-9c02-7ab1f88f044e)

Another big issue we had was our decoding algorithm didn’t work for when the student wrote on the barcode. To mitigate this, we modified our decoding algorithm to, instead of only checking if the pixels at the current x,y coordinate is black, checking if the ~20 pixels to the right are on average black (width of the bars is 20). This allowed the decoding algorithm to be robust against students writing on the barcode (unless they write in straight horizontal lines on the white lines of the barcode for some reason) and against noise (i.e white dots on the image). 
![image](https://media.github.iu.edu/user/18309/files/8fcf7902-52ab-4170-a9b5-afab901b9034)

However, when there was too much gaussian noise (with standard deviation > 50), our decoding algorithm failed. 
![image](https://media.github.iu.edu/user/18309/files/63ed1a0e-3582-4d33-8d2a-d9bcad57552f)

*Gaussian noise with standard deviation=80*

So, in extract.py, before decoding, we added a mean filter with a kernel size of 3 that looked at neighboring pixels values and made the current pixel black if their mean is less than a threshold value. We see that, even after this and after thresholding, we see some white noise inside the black bars:
![image](https://media.github.iu.edu/user/18309/files/e701137f-9951-4d78-a5a0-c9cce81830fd)
*White dots in bars*

So to remove the white dots, we also applied morphological ordering by applying a min filter, and then a max filter. This removed all of the white noise from the black bars successfully. We realized that the mean filter became redundant, so we removed it.
![image](https://media.github.iu.edu/user/18309/files/3d64d04a-bf23-479f-bda2-9ca045c15ea0)

*No white dots in bars due to min/max filtering*

Our decoding algorithm was able to extract the answers for injected.jpg with gaussian noise up until standard deviation of 80, after which accuracy of output files drops to 97 and below.

We also made our barcode work for when the barcode is not stuck to the left edge of the form, and we tested this by adding some padding to the left of the blank form (we saw extract.py works perfectly well, as we only start deciphering when we see the start block, i.e 50 consecutive black (on average) lines):

![image](https://media.github.iu.edu/user/18309/files/00344a02-8e18-405f-bc05-c82de37b34a1)

*Padding to the left of barcode doesn't effect decoding algorithm results*

### Limitations and Future
One of the assumptions we make is that our barcode can only be deciphered if the image size is the same as the original image (so after rescanning, the image size is the same). Since our barcode decoding algorithm accuracy depends on the exact pixel count of each bar being the same after scanning, it isn’t very robust to change in image size (downsizing, specifically height) as the height of the bars change. This is a tradeoff we had to make when we decided to decrease the difference between each consecutive answer width (now height) mapping from 3 to 1, so now we cannot allow for +-1 pixel error when comparing bar width to mapping. We upscale/downscale to the blank_form.jpg size the input image in extract.py to handle change in image size. Furthermore, if the scan is tilted, our extract.py fails to decode the bars. 

Our methods can be improved in a few ways. For one, since our method fails when there is tilt in the image, one could do spatial alignment between the image and a default image (original injected image before rescanning for example, which is blank_form.jpg) to un-tilt the image. Furthermore, if the rescanned image size is different, the bar pixel heights will be different, so having some kind of normalization that allows bar pixel heights to be transformed into the original injected.jpg bar pixel heights could allow it to work when image size is different. One way we could do this is by looking at how long the starting block (which is 50 pixels by default) is and dividing the height by 50 to get a factor that we could use to multiply the height mappings by. So, say the height of each bar is halved, then the starting bar will be 25 pixels, and so we divide the height mappings by 2 (so, “A” bar would have to be 3 pixels high instead of 6). Our original idea of having a bigger difference (of 3) between the height mappings/values would also mitigate this issue, but then you run the risk of having too large of a barcode on the page.

## Contributions of the Authors
### Sean: inject.py and extract.py, readme "How to run" section and "Injecting and Extracting" section
### Shreeja: Drawing and Highlighting Answers (Filled Box) in the image  in grade.py and change the code to take arguments according to the command. Also tested code on server
### Kundhana: Updated grade.py and created a text file named output.txt that has a quesion number, the marked options and x on the line if the student has written an answer to the left of the question number.







