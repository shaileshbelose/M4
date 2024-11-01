import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# Loading ready model
model = load_model('M4_v3.1.1.keras')

# Loading test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test_samples',
    target_size=(84, 84),
    batch_size=32,
    class_mode='categorical',  # or 'binary' if you have two classes
    shuffle=False
)

# Get the true labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Predict the labels
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Identify misclassified images
misclassified_indices = np.where(predicted_labels != true_labels)[0]

# Display misclassified images
for i in misclassified_indices:
    img_path = test_generator.filepaths[i]
    true_label = class_labels[true_labels[i]]
    predicted_label = class_labels[predicted_labels[i]]
    
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.show()

# Optional: Save misclassified images with labels for further analysis
misclassified_data = []
for i in misclassified_indices:
    img_path = test_generator.filepaths[i]
    true_label = class_labels[true_labels[i]]
    predicted_label = class_labels[predicted_labels[i]]
    misclassified_data.append((img_path, true_label, predicted_label))

# Save the misclassified data to a file
import csv
with open('misclassified_images.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Path', 'True Label', 'Predicted Label'])
    writer.writerows(misclassified_data)
