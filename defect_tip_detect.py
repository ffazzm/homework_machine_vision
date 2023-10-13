from cProfile import label
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from imutils import paths
import os
import cv2
import imutils

def detect_circle(filename):
    image = cv2.imread(filename)
    max_dimension = 800 
    image = imutils.resize(image, width=max_dimension)
    imageee = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circle from the end of tip
    circles = cv2.HoughCircles(
        # gray_blurred,          # Input image
        grayscale,          # Input image
        cv2.HOUGH_GRADIENT, # Detection method
        dp=1,                  # Inverse ratio of accumulator resolution to image resolution (1 for same resolution)
        minDist=5000,            # Minimum distance between detected circles
        param1=35,             # Higher threshold for Canny edge detection 35
        param2=100,             # Accumulator threshold for circle detection (lower values for more circles) 100
        minRadius=0,           # Minimum radius of detected circles
        maxRadius=0            # Maximum radius of detected circles (0 for automatic detection)
    )

    if circles is not None:
        # Convert circle coordinates to integers
        circles = np.uint16(np.around(circles))

        # Draw the detected circles on the original image
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # Create a contour for the circle
            contour_circle = np.array([[center[0] + radius, center[1]]])
            for angle in range(0, 360, 5):
                x = int(center[0] + radius * np.cos(np.radians(angle)))
                y = int(center[1] + radius * np.sin(np.radians(angle)))
                contour_circle = np.vstack((contour_circle, [[x, y]]))
                
            mask = np.zeros_like(imageee, dtype=np.uint8)

            cv2.circle(mask, center, int(radius+(radius*0.2)), (255, 255, 255), thickness=cv2.FILLED)
            mask = cv2.bitwise_not(mask)
            mask = cv2.subtract(imageee, mask)
            result_image = mask.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
            min_object_size = 25000  # Adjust this threshold as needed
            filtered_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > min_object_size:
                    filtered_contours.append(contour)
            
            for contour in filtered_contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                result_image = result_image[y:y+h, x:x+w]
    return result_image, circles

def prepare_data(X_image, X_feature):
    for img_path in X_image:
        img, circles = detect_circle(img_path)
        img = cv2.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # ignore blank images
        if circles is not None:
            feature = model.predict(x)
            X_feature.append(feature)
    return X_feature

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Extract Features
x = base_model.output
model = Model(inputs=base_model.input, outputs=x)
image_paths = paths.list_images("D:\\Python\\formulatrix_test\\D. Test Set - Vision Test\\test set\\train")
features = []
images_path = []
labels = []
for img_path in image_paths:  # Loop through your dataset
    images_path.append(img_path)
    labels.append(img_path.split(os.path.sep)[-2])

# Prepare Data for SVM
images_path = np.array(images_path)
labels = np.array(labels)  # Replace 'labels' with your actual labels

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(images_path, labels, test_size=0.2, random_state=42, stratify=labels)

X_feature_train = []
X_feature_test = []
prepare_data(X_train, X_feature_train)
X_feature_train = np.array(X_feature_train).reshape(len(X_feature_train), -1)

prepare_data(X_test, X_feature_test)
X_feature_test = np.array(X_feature_test).reshape(len(X_feature_test), -1)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.001, 0.01, 0.1]
}

# svm_classifier = svm.SVC()
# # Perform grid search with cross-validation
# stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# grid_search = GridSearchCV(svm_classifier, param_grid, cv=stratified_cv)
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_
# print(best_params)
# # Evaluate the best model on the test data
# accuracy = best_model.score(X_test, y_test)

# Result grid_search:
# C= 0.1, gamma= 0.001, kernel= 'poly'
# Train SVM Classifier
clf = svm.SVC(C= 0.1, gamma= 0.001, kernel= 'poly')
clf.fit(X_feature_train, y_train)

# Evaluate SVM Model
accuracy = clf.score(X_feature_test, y_test)
print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

y_pred = clf.predict(X_feature_test)
report = classification_report(y_test, y_pred)
# Print the classification report
print(report)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, y_pred)
import pandas as pd
confusion_df = pd.DataFrame(confusion, index=np.unique(labels), columns=np.unique(labels))
print(confusion_df)

# show predicted labels
X_feature_test = []
for img_path_test in X_test:
    img, circles = detect_circle(img_path_test)
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    X_feature_test.append(feature)
    X_feature_test = np.array(X_feature_test).reshape(len(X_feature_test), -1)
    y_pred = clf.predict(X_feature_test)
    # print(y_pred)
    cv2.putText(img, y_pred[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    X_feature_test = []


