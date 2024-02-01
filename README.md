# Face Shape Detection and Style Recommendations


## Introduction:


In an era where technological innovations continually reshape the landscape of user experiences, the fusion of artificial intelligence and fashion has opened new frontiers. This project endeavors to revolutionize the eyewear selection process by integrating cutting-edge computer vision techniques and machine learning algorithms.
The aim is to provide users with a personalized and seamless eyewear recommendation system, incorporating factors such as face shape, skin tone, and frame colors. Leveraging the power of face detection algorithms like dlib and OpenCV, the system initiates the process by categorizing the user's face shape, laying the foundation for tailored eyewear suggestions. The lens frame overlay feature, meticulously aligned with facial landmarks, allows users to virtually try on different frames, enhancing their ability to make informed choices.
Facial landmark detection, particularly at the glabella point, serves as a gateway to understanding individual skin tones, enabling the system to offer precise recommendations that resonate with the user's unique characteristics. The culmination of these elements results in a sophisticated and user-centric eyewear recommendation system, poised to redefine how individuals explore and select eyewear that complements their distinct facial attributes and personal style.


## Features:

### Face Shape Detection:
Utilize a face detection algorithm to identify and determine the user's face shape. This can be achieved using popular libraries like dlib, OpenCV, or similar tools.

### Lens Frame Overlay:
After detecting the face shape, overlay the lens frame image provided by the user onto their facial image.Ensure that the frame image is accurately positioned by aligning it with the 37th and 46th facial landmarks.Resize the frame image as needed to fit perfectly within these reference points.

### Glabella Point and Skin Tone Detection:
Employ the dlib library or a similar facial landmark detection method to locate the glabella point (the point between the eyebrows).Extract the RGB color code of this specific point on the user's facial image.

### Skin Tone Assessment:
 Use the extracted RGB color code to determine the user's skin tone. You can compare it to predefined skin tone color ranges or use machine learning algorithms to classify it.

### Color Recommendation:
Based on the user's skin tone classification, provide lens frame color recommendations that complement their skin tone. You can create a lookup table or use algorithms to make these recommendations.

    
## Requirements
### Hardware Requirements
    Computer or Laptop:
### Softare Requirements

#### Operating System:  
Windows, macOS, or Linux.
#### Development Environment:  
Visual Studio Code, PyCharm, or others.
#### Programming Language: 
Python, and relevant libraries and frameworks.
#### Image Processing Libraries: 
Integration of image processing libraries like OpenCV for tasks such as blurring, unblurring, and image manipulation.
#### Text Recognition Library: 
Integration of OCR (Optical Character Recognition) libraries such as Tesseract to accurately recognize and extract text from images.
#### Dlib Library:
One of the primary uses of dlib is for facial landmark detection. It provides tools to identify and locate key points on a face, enabling applications such as face recognition, expression analysis, and augmented reality.


## Program
```python

import cv2
import dlib
import numpy as np
import face_recognition
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Load the input image
image = cv2.imread('IMG_20210113_084045.jpg')
# Detect the face landmarks in the input image
face_landmarks_list = face_recognition.face_landmarks(image)
# Convert the image to grayscale (dlib face detector works on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Loop over the detected faces
for face in faces:
    # Extract the coordinates of the bounding box for each face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Calculate the desired top point of the bounding box (e.g., where the hairline ends)
    hairline_y = int(y - 0.2 * h)  # Adjust the multiplier (0.2) as needed

    # Adjust the bounding box to include the forehead region
    h = int(h + (y - hairline_y))
    forehead_start= hairline_y
# Access the 9th point of the chin for the first face detected in the input image
chin_9th_point = face_landmarks_list[0]['chin'][8]
chin_x = chin_9th_point[0]
chin_y = chin_9th_point[1]
# Calculate the distance between the forehead start point and chin point
facelength = abs(chin_9th_point[1] - forehead_start)

# Extract the facial landmarks for the first face detected
landmarks = predictor(gray, face)

# Detect the face landmarks in the input image
faces = detector(image)
landmarks = predictor(image, faces[0])

# Define the eyebrow landmark indices
left_eyebrow_indices = [17]
right_eyebrow_indices = [25]

# Calculate the distance between the left and right eyebrow landmarks
left_x, left_y = landmarks.part(17).x, landmarks.part(17).y
right_x, right_y = landmarks.part(25).x, landmarks.part(25).y
forehead = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)


# Calculate the distance between the left and right eyebrow landmarks
left_x, left_y = landmarks.part(17).x, landmarks.part(17).y
right_x, right_y = landmarks.part(25).x, landmarks.part(25).y
distance = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)
# Get the coordinates of the 10th and 13th Facial points
point_10 = face_landmarks_list[0]['chin'][9]
point_14 = face_landmarks_list[0]['chin'][12]

# Compute the Euclidean distance between the two points
jawlength = ((point_10[0]-point_14[0])**2 + (point_10[1]-point_14[1])**2)**0.5
# Extract the facial landmarks for the first face in the input image
chin_points = face_landmarks_list[0]['chin']

# Get the 8th and 10th points of the chin
point_8 = chin_points[7]
point_10 = chin_points[9]

# Compute the Euclidean distance between the 8th and 10th points of the chin
chinlength = math.sqrt((point_8[0]-point_10[0])**2 + (point_8[1]-point_10[1])**2)
#Extract the facial landmarks for the first face in the input image
chin_points = face_landmarks_list[0]['chin']

# Get the 2th and 16th points of the chin
point_2 = chin_points[1]
point_16 = chin_points[15]

# Compute the Euclidean distance between the 2th and 16th points of the chin
cheekbone = math.sqrt((point_2[0]-point_16[0])**2 + (point_2[1]-point_16[1])**2)
def calculate_ratios(facelength, cheekbone, forehead, jawlength, chinlength):
    cheekbone_ratio = cheekbone / facelength
    jawline_ratio = jawlength / facelength
    forehead_ratio = forehead / facelength
    chin_ratio = chinlength / facelength

    return cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio


def determine_face_shape(cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio):
    if forehead_ratio > 0.27:
        if chin_ratio < 0.18:
            if cheekbone_ratio > 0.51:
                return "Heart-shaped"
            else:
                return "Oval"
        else:
            if cheekbone_ratio > 0.47 and jawline_ratio > 0.37:
                return "Square"
            elif cheekbone_ratio > 0.47 and jawline_ratio < 0.36:
                return "Diamond"
            else:
                return "Triangle"
    else:
        if cheekbone_ratio < 0.48:
            if jawline_ratio < 0.36:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    return "Soft"
            else:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    return "Square"
        else:
            if jawline_ratio < 0.36:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    return "Soft"
            else:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    if forehead_ratio > 0.50:
                        return "Oblong"
                    else:
                        return "Square"

cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio = calculate_ratios(facelength, cheekbone, forehead, jawlength, chinlength)


face_shape = determine_face_shape(cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio)
print("Face shape:", face_shape)

# Detect faces in the original image
faces = dlib.get_frontal_face_detector()(image)

# Assuming there's only one face in the image, you can modify this for multiple faces
face = faces[0]

# Function to calculate the midpoint between two points
def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

# Function to calculate the distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Landmark indices for eyes
left_eye_midpoint1 = midpoint((landmarks.part(0).x, landmarks.part(0).y), (landmarks.part(36).x, landmarks.part(36).y))
left_eye_midpoint2 = midpoint((landmarks.part(45).x, landmarks.part(45).y), (landmarks.part(16).x, landmarks.part(16).y))

right_eye_midpoint1 = midpoint((landmarks.part(37).x, landmarks.part(37).y), (landmarks.part(19).x, landmarks.part(19).y))
right_eye_midpoint2 = midpoint((landmarks.part(24).x, landmarks.part(24).y), (landmarks.part(44).x, landmarks.part(44).y))

# Points to define the rectangle
upper_left_point = (left_eye_midpoint1[0], right_eye_midpoint2[1])
upper_right_point = (left_eye_midpoint2[0], right_eye_midpoint2[1])
lower_left_point = (left_eye_midpoint1[0], landmarks.part(28).y)  # Use the 28th point for the bottom left
lower_right_point = (landmarks.part(28).x, landmarks.part(28).y)  # Use the 28th point for the bottom right

# Print the coordinates of the four points
print("Upper Left Point:", upper_left_point)
print("Upper Right Point:", upper_right_point)
print("Lower Left Point:", lower_left_point)
print("Lower Right Point:", lower_right_point)

# Calculate the length and breadth of the rectangle
length = distance(upper_left_point, upper_right_point)
breadth = distance(upper_left_point, lower_left_point)

# Print the length and breadth of the rectangle
print("Length of the Rectangle:", length)
print("Breadth of the Rectangle:", breadth)

# Load the PNG image to paste
image_to_paste_path = "glasse.png"
image_to_paste = Image.open(image_to_paste_path).convert("RGBA")  # Convert to RGBA if not already

# Resize the image to fit into the bounding box
image_to_paste_resized = image_to_paste.resize((int(length), int(breadth)))

# Convert the resized image to a NumPy array
image_to_paste_array = np.array(image_to_paste_resized)

# Create a figure and axis
fig, ax = plt.subplots()

# Display the original image
ax.imshow(image)

# Draw lines and plot points
ax.plot([upper_left_point[0], upper_right_point[0]], [upper_left_point[1], upper_right_point[1]], color='red')

# Draw a rectangle around the eyes
eye_rect = patches.Rectangle(upper_left_point, length, breadth, linewidth=2, edgecolor='blue', facecolor='none')

# Add the rectangle to the axis
ax.add_patch(eye_rect)

# Create a mask using the alpha channel of the pasted image
alpha_mask = image_to_paste_array[:, :, 3] / 255.0

# Paste the resized image onto the original image using alpha blending
original_image_copy = image.copy()
original_image_copy[upper_left_point[1]:lower_left_point[1], upper_left_point[0]:upper_right_point[0], :3] = (
    original_image_copy[upper_left_point[1]:lower_left_point[1], upper_left_point[0]:upper_right_point[0], :3] * (1 - alpha_mask[:, :, None])
    + image_to_paste_array[:, :, :3] * alpha_mask[:, :, None]
)

# Show the image with the rectangle around the eyes and the pasted image
plt.axis('off')
plt.imshow(original_image_copy)
plt.show()

def extract_skin_undertone(hex_code):
    # Convert hex to RGB
    r, g, b = int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:7], 16)

    # Calculate the difference between channels
    rg_diff = abs(r - g)
    rb_diff = abs(r - b)

    # Classify undertone based on channel differences
    if r > g and r > b:
        if rg_diff < 15 and rb_diff < 15:
            return "Neutral undertone"
        elif g > b:
            return "Warm undertone"
        else:
            return "Cool undertone"
    elif g > r and g > b:
        if rg_diff < 15 and rb_diff < 15:
            return "Neutral undertone"
        elif r > b:
            return "Cool undertone"
        else:
            return "Warm undertone"
    elif b > r and b > g:
        if rg_diff < 15 and rb_diff < 15:
            return "Neutral undertone"
        elif r > g:
            return "Cool undertone"
        else:
            return "Warm undertone"
    else:
        return "Neutral undertone"

def extract_glabella_color(image_path):
    # Load the facial landmarks model from dlib
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Read the image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    faces = detector(image_rgb)

    if len(faces) > 0:
        # Assuming there's only one face in the image, you can modify this for multiple faces
        face = faces[0]

        # Get the facial landmarks for the detected face
        landmarks = predictor(image_rgb, face)

        # Extract the x and y coordinates of the glabella point (midpoint between 21 and 22)
        x_glabella = (landmarks.part(21).x + landmarks.part(22).x) // 2
        y_glabella = (landmarks.part(21).y + landmarks.part(22).y) // 2

        # Display the skin color at the glabella point
        hex_code, skin_color_rgb = extract_skin_color_at_location(image_path, (x_glabella, y_glabella))
        print("Skin Color at Glabella Point Hex Code:", hex_code)

        # Determine skin undertone
        undertone = extract_skin_undertone(hex_code)
        print("Skin Undertone:", undertone)

        # Display the skin color at the glabella point
        plt.imshow([[skin_color_rgb]])
        plt.title("Skin Color at Glabella Point")
        plt.axis("off")
        plt.show()

    else:
        print("No face detected in the image.")

def extract_skin_color_at_location(image_path, location):
    # Read the image
    image = cv2.imread(image_path)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get the skin color at the specified location
    skin_color_at_location = hsv[location[1], location[0]]

    # Convert HSV to BGR for display
    skin_color_bgr = cv2.cvtColor(np.uint8([[skin_color_at_location]]), cv2.COLOR_HSV2BGR)[0][0]

    # Convert BGR to RGB for displaying with matplotlib
    skin_color_rgb = cv2.cvtColor(np.uint8([[skin_color_bgr]]), cv2.COLOR_BGR2RGB)[0][0]

    # Convert BGR to hexadecimal
    hex_code = "#{:02x}{:02x}{:02x}".format(skin_color_bgr[2], skin_color_bgr[1], skin_color_bgr[0])

    return hex_code, skin_color_rgb

# Example usage:
image_path='IMG_20210113_084045.jpg'
extract_glabella_color(image_path)



```


## Output

### Lens

![Screenshot (40)](https://github.com/Loghul1722/project1/assets/132638997/81760f1b-933f-4e5f-99a8-aa960f21022f)


### Skin Tone


![Screenshot (41)](https://github.com/Loghul1722/project1/assets/132638997/29a6eac3-2541-4fc5-a518-3f084aa7f1ae)


## Result


This project innovative fashion transformation project, driven by thecapabalities of artificial intelligence and computer vision, is poised to revolutionize the way individuals approach personal style.By providing personalized recommendations, it empowers users to elevate their fashion game with confidence.
As the fashion industry evolves,this project leads the way towards a more personalized,informed, and technologically integrated future for fashion enthusiasts worldwide.
