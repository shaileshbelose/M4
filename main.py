# main.py

import tensorflow as tf
from model import build_model
from data import get_data_generators

# Define paths
train_dir = 'train_samples'
test_dir = 'test_samples'
num_classes = 33
img_size = (84, 84)
batch_size = 32

# Get data generators
train_dataset, validation_dataset = get_data_generators(train_dir, test_dir, img_size, batch_size, num_classes)

# Build model
model = build_model(num_classes)

# Compile model with L2 regularization
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# Model checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'M4_v3.1.1.keras', 
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_dataset,
    epochs=40,
    validation_data=validation_dataset,
    callbacks=[reduce_lr, checkpoint, early_stopping]
)

# Evaluate model
test_loss, test_acc = model.evaluate(validation_dataset)
print(f'Test accuracy: {test_acc}')

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
