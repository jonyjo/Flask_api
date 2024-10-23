import cv2
import numpy as np
import os
#import mediapipe as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
from scipy.signal import savgol_filter
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM, Dropout
from keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import img_to_array,load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import os
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.models import Model
from keras.layers import Input,concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7, InceptionResNetV2, InceptionV3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image


app = Flask(__name__)


def CC_LSTM_model(feature_shape, input_shape=(10, 224, 224, 3),
                  n_classes=8, 
                  learning_rate=0.0001, 
                  lstm_units=256, 
                  dropout_rate=0.5):

    # Define image input for original images
    image_input = Input(shape=input_shape)
    
    # Define feature input for extracted features
    feature_input = Input(shape=feature_shape)

    # CNN layers on image input
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(image_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    # Flatten the output of the CNN
    x = TimeDistributed(Flatten())(x)

    # LSTM layer
    x = LSTM(lstm_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)

    # Combine with feature input
    x = concatenate([x, feature_input])  # Merge the LSTM output with the features

    # Fully connected layer
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(dropout_rate)(x)

    # Output layer for classification
    output = Dense(n_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)

    # Create the model
    model = Model(inputs=[image_input, feature_input], outputs=output)

    # Compile the model with Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

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
        
def load_features_images(feat_folder, img_folder, target_size=(224, 224)):
    """
    Load features from .npy files in the specified folder.
    Returns a list of feature arrays and their corresponding identifiers.
    Each identifier corresponds to a single image.
    """
    labels = []
    features = []
    identifiers = []  # Store the identifiers for matching with images

    for filename in sorted(os.listdir(feat_folder)):
        if filename.endswith('.npy'):
            feature_data = np.load(os.path.join(feat_folder, filename))
            features.append(feature_data)  # Append the feature array
            
            # Extract the identifier (without the .npy extension)
            identifier = filename.replace('.npy', '')
            identifiers.append(identifier)
            labels.append(identifier.split(' ')[0])
            
    # Load corresponding images as individual samples
    image_sequences = []
    for identifier in identifiers:
        img_filename = f"{identifier}.jpg"  # Modify this if your image naming convention is different
        img_path = os.path.join(img_folder, img_filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)  # Load image
            img = cv2.resize(img, target_size)  # Resize to target size
            img = img.astype('float32') / 255.0
            image_sequences.append(img)  # Append single image instead of a sequence
            
    if not image_sequences:  # Return empty array if no images found
        print("No images found for the given identifiers.")
    
    return np.array(features), labels, np.array(image_sequences)

    
def segment_image_kmeans(img, num_clusters=9):
    # Load the image
    #img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
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



# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features
def extract_features(img):
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array,verbose=False)
    return features




# Step 2: Feature Extraction using Pre-trained Models
def load_models():
    efficientnet_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')
    inception_resnet_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    googlenet_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    return efficientnet_model, inception_resnet_model, googlenet_model

def extract_features_grd(image, models):
    image = img_to_array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)

    features = []
    for model in models:
        feature = model.predict(image,verbose=False)
        features.append(feature)

    return np.concatenate(features, axis=1)  # Combine features from all models
@app.route('/Prepare_data',methods=['POST'])
def Prepare_data():
    aug_count=1
    dataset_folder = request.json.get('data_folder')
    img_folder = request.json.get('img_folder')
    feat_folder = request.json.get('feat_folder')
    # Define the augmentation generator using Keras
    datagen = ImageDataGenerator(
        rotation_range=30,          # Randomly rotate images by 30 degrees
        width_shift_range=0.1,      # Shift the width of the image by 10%
        height_shift_range=0.1,     # Shift the height of the image by 10%
        shear_range=0.2,            # Shear intensity
        zoom_range=0.2,             # Zoom into the image
        horizontal_flip=True,       # Randomly flip images horizontally
        fill_mode='nearest'         # Fill in newly created pixels after augmentations
    )
    # Get the total number of images (assuming all files with .jpg extension are images)
    num_images = len([filename for filename in os.listdir(dataset_folder) if filename.endswith(".jpg")])

    # Create a progress bar using tqdm
    with tqdm(total=num_images, desc="Preprocessing and Extracting Features") as progress_bar:
          
        for filename in os.listdir(dataset_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(dataset_folder, filename)
                img = cv2.imread(image_path)

                # Segment the image
                seg_img = segment_image_kmeans(img)
                
                # Step 1: Remove background
                preprocessed_img = remove_bg(seg_img)
    
                # Save the original preprocessed image with background removed
                output_img_path = f"{img_folder}/{filename.split('.')[0]}.jpg"
                cv2.imwrite(output_img_path, preprocessed_img)
    
                # Step 2: Augment the preprocessed image
                # Convert the image to the required format (4D array for the ImageDataGenerator)
                img_array = np.expand_dims(preprocessed_img, axis=0)  # Shape: (1, height, width, channels)
    
                # Generate augmented images and save them
                aug_index = 0
                for batch in datagen.flow(img_array, batch_size=1):
                    aug_img = batch[0].astype('uint8')  # Convert to uint8 format for saving
                    
                    # Save augmented images
                    aug_filename = f"{filename.split('.')[0]}_aug{aug_index}.jpg"
                    aug_img_path = os.path.join(img_folder, aug_filename)
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Step 3: Extract features from augmented images
                    features = extract_features(aug_img)
                    # Save extracted features
                    np.save(os.path.join(feat_folder, aug_filename.replace('.jpg', '.npy')), features)
                    
                    aug_index += 1
                    if aug_index >= aug_count:  # Limit to 'aug_count' augmentations
                        break
    
                # Step 4: Extract features from the original preprocessed image
                original_features = extract_features(preprocessed_img)
                np.save(os.path.join(feat_folder, filename.replace('.jpg', '.npy')), original_features)        
                # Update the progress bar after each image is processed
                progress_bar.update(1)
def get_fruit_parameters(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to grayscale for better contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0, (0, 0, 0)  # Return zeros if no contours found

    # Get the largest contour which is assumed to be the fruit
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate length and size
    length = h / 10  # Assuming height in pixels can be converted to cm
    size = (w * h) / 100  # Area in pixels converted to a rough estimate in square cm

    # Calculate average color
    mean_color = cv2.mean(img[y:y+h, x:x+w])[:3]  # Mean color within the bounding box
    mean_color = (mean_color[2], mean_color[1], mean_color[0])  # Convert to RGB

    return length, size, mean_color



# Define training API
@app.route('/training', methods=['POST'])
def training():
    # Get the required folders from the request data
    data_folder = request.json.get('data_folder')
    img_folder = request.json.get('img_folder')
    feat_folder = request.json.get('feat_folder')
    data=[]
    
    # Define grading rules (example thresholds)
    grading_rules = {
        'Banana': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]},  
        'Apple': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]},  
        'Custard_apple': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]}, 
        'Lime': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]},  
        'Orange': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]}, 
        'Pomegranate': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]}, 
        'Guava': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]},  
        'Mango': {'TYPE-I': [0.33, 0.66], 'TYPE-II': [0.66, 1], 'TYPE-III': [0, 0.33]}, 
    }
    
    # Load features and identifiers
    X_features, labels,X_images= load_features_images(feat_folder,img_folder)
    
    
    
    # Reshape the features to (num_samples, 2048)
    X_features = X_features.reshape(X_features.shape[0], -1)  # Remove the second dimension
    
    # Check reshaped feature dimensions
    print(f'Reshaped Features shape: {X_features.shape}')  # Expected shape: (num_samples, 2048)
    
    
    # Step 1: Encode string labels into integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # Convert strings to integers
    n_classes = len(label_encoder.classes_)
    y_labels = to_categorical(encoded_labels, num_classes=n_classes)
    
    # Define the model (adjust feature_shape as per your feature extraction)
    num_features = X_features.shape[1]  # This should match the number of features
    model = CC_LSTM_model(feature_shape=(num_features,),input_shape=(1, 224, 224, 3),  n_classes=n_classes)
    X_train_img, X_test_img,X_train_feat,X_test_feat, y_train, y_test = train_test_split(X_images,X_features, y_labels,test_size=0.2,random_state=21)
    # Train the model
    X_train_img = np.expand_dims(X_train_img, axis=1)  # Shape becomes (num_samples, 1, 224, 224, 3)
    X_test_img = np.expand_dims(X_test_img, axis=1)    # Shape becomes (num_samples, 1, 224, 224, 3)

    model.fit([X_train_img, X_train_feat], y_train, batch_size=16, epochs=1,validation_split=0.2)
    
    
    # Save the classification model
    model.save('classification_model.h5')
    
    for img in os.listdir(data_folder):
            img_path=os.path.join(data_folder,img)
            # Extract parameters for grading
            length, size, color = get_fruit_parameters(img_path)
    
            # Determine grade based on length (you can also include other parameters)
            if length >= grading_rules[f'{img.split(" ")[0]}']['TYPE-I'][0] and length <= grading_rules[f'{img.split(" ")[0]}']['TYPE-I'][1]:
                grade = 'TYPE-I'
            elif length >= grading_rules[f'{img.split(" ")[0]}']['TYPE-II'][0] and length <= grading_rules[f'{img.split(" ")[0]}']['TYPE-II'][1]:
                grade = 'TYPE-II'
            else:
                grade = 'TYPE-III'
    
            data.append({'image_path': img_path, 'length': length, 'size': size, 'color': color, 'grade': grade})
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    
    # Load models
    models = load_models()
    
    # Extract features for all images
    features_list = []
    # Get the total number of images (assuming all files with .jpg extension are images)
    num_images = len(df)
    
    # Create a progress bar using tqdm
    with tqdm(total=num_images, desc="Extracting Features for grade classification model") as progress_bar:
          
        for img_path in df['image_path']:
            image = load_img(img_path, target_size=(224, 224))  # Resize to fit model
            features = extract_features_grd(image, models)
            features_list.append(features)
            progress_bar.update(1)
    
    # Convert to array
    X_features = np.array(features_list)
    X_features = X_features.reshape(X_features.shape[0], -1)  # Flatten to 2D
    # Step 3: Normalize Features
    scaler = MinMaxScaler()
    X_features_normalized = scaler.fit_transform(X_features)
    
    # Encode labels
    grade_label_encoder = LabelEncoder()
    encoded_labels = grade_label_encoder.fit_transform(df['grade'])
    y_labels = to_categorical(encoded_labels)  # Convert to one-hot encoding
    # Step 4: Create Model
    def create_model(input_shape, num_classes):
        inputs = Input(shape=(input_shape,))
        x = Dense(512, activation='relu')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
    
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    input_shape = X_features_normalized.shape[1]  # Number of features from models
    num_classes = y_labels.shape[1]  # Number of unique grading classes
    grading_model = create_model(input_shape, num_classes)
    
    # Step 5: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X_features_normalized, y_labels, test_size=0.2, random_state=42)
    
    # Step 6: Train the Model
    grading_model.fit(X_train, y_train, batch_size=4, epochs=20, validation_split=0.2)
    
    # Save the grading model
    grading_model.save('grading_model.h5')  # Save the model in HDF5 format
    # Step 7: Evaluate the Model
    loss, accuracy = grading_model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    return jsonify({"message": "Training completed and model saved."})

    
@app.route('/testing', methods=['POST'])
def testing():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    class_names = ['Apple', 'Banana', 'Custard_apple', 'Guava', 'Lime', 'Mango', 'Orange', 'Pomegranate']

    # Load your models
    classification_model = load_model('classification_model.h5')
    grading_model = load_model('grading_model.h5')

    # Load image from the uploaded file and convert to NumPy array
    image = Image.open(file.stream)
    image_np = np.array(image)  # Convert PIL image to NumPy array

    # Ensure the image has 3 channels
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:  # RGBA image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    img = cv2.resize(image_np, (224, 224))  # Resize to target size
    img = img.astype('float32') / 255.0

    segm_img = segment_image_kmeans(image_np)  # Segment the image
    pro_img = remove_bg(segm_img)  # Process the image
    feas = extract_features(pro_img)  # Extract features
    image_batch = np.expand_dims(img, axis=0)  # Shape (1, 224, 224, 3)
    image_batch = np.expand_dims(image_batch, axis=1)  # Shape (1, 1, 224, 224, 3)


    # Make predictions
    classification_result = classification_model.predict([image_batch, feas])
    # Load models
    models = load_models()
    image_grd=extract_features_grd(image_np,models)
    grading_result = grading_model.predict(image_grd)  # Add a new axis for batch size

    # Return the result as JSON
    return jsonify({
        "classification": classification_result.tolist(),
        "grading": grading_result.tolist()
    })

# data_folder="../Demo"
# img_folder="./demo1"
# feat_folder="./demo2"
# main(data_folder,img_folder,feat_folder)

#if __name__ == '__main__':
#    # Get the PORT from environment variables, default to 5000 if not set
#    port = int(os.environ.get('PORT', 5000))
#    # Bind to 0.0.0.0 and listen on the specified port
#    app.run(host='0.0.0.0', port=port)
