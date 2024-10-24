import cv2
import numpy as np
from scipy.signal import savgol_filter

# Set your API token here
api_token = "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"
import requests


# The URL of your Hugging Face model
model_url = "https://api-inference.huggingface.co/models/Jo787jo/Fruits_model"

headers = {
    "Authorization": f"Bearer {api_token}",
}

image_data=cv2.imread("C:/Users/hello/Downloads/Jony/JO/Sobha_patel/Project_sobha_patel/Dataset/Apple (1).jpg")
def remove_bg(imgOg):
    #img = cv2.imread(imgOg)
    blurred = cv2.GaussianBlur(imgOg, (5, 5), 0) # Remove noise
    
    # Function for edge detection using Sobel operator
    def edgedetect(channel):
        # Apply Sobel operator in X direction
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    
        # Apply Sobel operator in Y direction
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    
        # Calculate the gradient magnitude
        sobel = np.hypot(sobelX, sobelY)
    
        # Clip values to the range [0, 255]
        sobel[sobel > 255] = 255
    
        # Convert back to uint8
        return np.uint8(sobel)
    
    # Apply edge detection to each color channel (R, G, B)
    edgeImg = np.max(np.array([edgedetect(blurred[:,:, 0]), 
                               edgedetect(blurred[:,:, 1]), 
                               edgedetect(blurred[:,:, 2])]), axis=0)
    
    
    mean = np.mean(edgeImg);
    # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0;
    
    def findSignificantContours(img, edgeImg):
        # Find contours in the edge image (OpenCV 4.x returns 2 values)
        contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # Find level 1 contours
        level1 = []
        for i, tupl in enumerate(hierarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)
    
        # Find contours with large surface area
        significant = []
        tooSmall = edgeImg.size * 4 / 100  
        for tupl in level1:
            contour = contours[tupl[0]]
            area = cv2.contourArea(contour)
            if area > tooSmall:
                significant.append([contour, area])
    
                # Draw the significant contour on the original image
                cv2.drawContours(img, [contour], 0, (0, 255, 0), 1, cv2.LINE_AA, maxLevel=1)
    
        # Sort significant contours by area
        significant.sort(key=lambda x: x[1], reverse=True)
    
        return [x[0] for x in significant]  # Return the contours
    
    # Convert edge image to uint8 type
    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    
    # Find significant contours
    significant = findSignificantContours(imgOg, edgeImg_8u)
    

    # Create a mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    
    # Invert the mask
    mask = np.logical_not(mask)
    
    # Remove the background
    imgOg[mask] = 0
    
    # Select a specific contour (for example, the largest one)
    if len(significant) > 0:
        contour = significant[0]  # The largest contour, as they are sorted by area 
        # Approximate the contour using arcLength
        epsilon = 0.10 * cv2.arcLength(contour, True)  # 10% of the contour perimeter    
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
        
        # Use Savitzky-Golay filter to smooth the contour
        window_size = int(round(min(imgOg.shape[0], imgOg.shape[1]) * 0.05))  # 5% of the image     dimensions
        if window_size % 2 == 0:
            window_size += 1  # Ensure the window size is odd
    
        # Extract x and y points of the contour for smoothing
        x = savgol_filter(contour[:, 0, 0], window_size, 3)
        y = savgol_filter(contour[:, 0, 1], window_size, 3)
    
        # Construct the new smoothed contour
        approx_smoothed = np.empty((x.size, 1, 2), dtype=np.int32)
        approx_smoothed[:, 0, 0] = x
        approx_smoothed[:, 0, 1] = y
    
        # Draw the smoothed contour on the image
        cv2.drawContours(imgOg, [approx_smoothed], 0, (255, 0, 0), 1)  # Draw the smoothed     contour in blue
        imgBg=cv2.cvtColor(imgOg, cv2.COLOR_BGR2RGB)
        return imgBg
      
    else:
        print("No significant contours found.")
        return None

def segment_image_kmeans(image_path, num_clusters=9):
    # Load the image
    img = cv2.resize(image_path, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying

    # Reshape the image to be a list of pixels
    pixels = img.reshape((-1, 3))

    # Convert to float32 for K-means
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and create segmented image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    return segmented_image


# Segment the image
seg_img = segment_image_kmeans(image_data)

# Step 1: Remove background
preprocessed_img = remove_bg(seg_img)

img_array = np.expand_dims(preprocessed_img, axis=0)  # Shape: (1, height, width, channels)

# Assuming img_array is a NumPy array
img_array = img_array.astype(np.uint8)  # Ensure the array type is correct
img_bytes = img_array.tobytes()  # Convert to bytes

# Send a request to the Inference API
response = requests.post(model_url, headers=headers, data=img_bytes)

# Check the response
if response.status_code == 200:
    prediction = response.json()
    print("Prediction:", prediction)
else:
    print(f"Error {response.status_code}: {response.text}")