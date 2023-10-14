try :
    import os, cv2, imutils
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn import svm
    from imutils import paths
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
except ImportError:
    import subprocess

    def install(package):
        subprocess.check_call(["pip", "install", package])

    # List of required packages
    required_packages = [
        "numpy", "tensorflow", "scikit-learn", "imutils", "opencv-python", "imutils", "pandas" 
    ]

    for package in required_packages:
        print(f"{package} is not installed. Installing now...")
        install(package)
        print(f"Successfully installed {package}.")
        
    import os, cv2, imutils
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn import svm
    from imutils import paths
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

def detect_circles(filename):
    """
    Detects circles in an image and returns the resulting image and the coordinates of the circles.

    Args:
        filename (str): Path to the image file.

    Returns:
        tuple: A tuple containing the resulting image and the coordinates of the circles.

    """
    # Load the image
    image = cv2.imread(filename)

    # Resize the image to a maximum dimension of 800
    max_dimension = 800
    image = imutils.resize(image, width=max_dimension)

    # Create a copy of the image
    image_copy = image.copy()

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        grayscale, cv2.HOUGH_GRADIENT, dp=1, minDist=5000, param1=35, param2=100, minRadius=0, maxRadius=0
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

            # Create a mask for the circle
            mask = np.zeros_like(image_copy, dtype=np.uint8)
            cv2.circle(mask, center, int(radius + (radius * 0.2)), (255, 255, 255), thickness=cv2.FILLED)
            mask = cv2.bitwise_not(mask)
            mask = cv2.subtract(image_copy, mask)
            result_image = mask.copy()

            # Convert the mask to grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Threshold the mask to obtain a binary mask
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on minimum object size
            min_object_size = 25000
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_object_size]

            # Crop the result image based on the filtered contours
            for contour in filtered_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                result_image = result_image[y:y + h, x:x + w]

    return result_image, circles

def prepare_data(X_image, X_feature):
    """
    Preprocesses the images in X_image, extracts features using a pre-trained VGG16 model, and appends the features to X_feature.

    Args:
        X_image (list): List of image paths.
        X_feature (list): List to store the extracted features.

    Returns:
        list: Updated X_feature list with the extracted features.
    """

    for img_path in X_image:
        # Detect circles in the image
        img, circles = detect_circles(img_path)

        # Resize image to (224, 224)
        img = cv2.resize(img, (224, 224))

        # Convert image to array
        x = tf.keras.preprocessing.image.img_to_array(img)

        # Expand dimensions to match the expected input shape of VGG16 model
        x = np.expand_dims(x, axis=0)

        # Preprocess input image to match the format required by VGG16 model
        x = tf.keras.applications.vgg16.preprocess_input(x)

        # Ignore blank images
        if circles is not None:
            # Extract features using VGG16 model
            feature = model.predict(x)

            # Append the features to X_feature
            X_feature.append(feature)

    return X_feature

# Load VGG16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
features = base_model.output
model = tf.keras.models.Model(inputs=base_model.input, outputs=features)

image_paths_list = paths.list_images("D:\\Python\\formulatrix_test\\D. Test Set - Vision Test\\test set\\train")
image_paths_array = []
image_labels = []
for image_path in image_paths_list:
    image_paths_array.append(image_path)
    image_labels.append(image_path.split(os.path.sep)[-2])

# Prepare Data for SVM
image_paths_array = np.array(image_paths_array)
image_labels = np.array(image_labels)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(image_paths_array, image_labels, test_size=0.2, random_state=42, stratify=image_labels)

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

# Perform grid search with cross-validation
# svm_classifier = svm.SVC()
# stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(svm_classifier, param_grid, cv=stratified_cv)
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Evaluate the best model on the test data
# accuracy = best_model.score(X_test, y_test)

# Train SVM Classifier
clf = svm.SVC(C=0.1, gamma=0.001, kernel='poly')
clf.fit(X_feature_train, y_train)

# Evaluate SVM Model
accuracy = clf.score(X_feature_test, y_test)
print("Accuracy:", accuracy)

# Print the classification report
y_pred = clf.predict(X_feature_test)
report = classification_report(y_test, y_pred)
print(report)

# Print the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(confusion, index=np.unique(image_labels), columns=np.unique(image_labels))
print(confusion_df)

# show predicted labels
X_feature_test = []
for img_path_test in X_test:
    img, circles = detect_circles(img_path_test)
    img = cv2.resize(img, (224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    feature = model.predict(x)
    X_feature_test.append(feature)
    X_feature_test = np.array(X_feature_test).reshape(len(X_feature_test), -1)
    y_pred = clf.predict(X_feature_test)
    cv2.putText(img, y_pred[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    X_feature_test = []
