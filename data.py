# data.py

import tensorflow as tf
import os

def preprocess_image(image):
    image = tf.image.resize(image, [84, 84])
    image = image / 255.0  # Normalize to [0,1] range
    return image

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image)
    return image, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def get_dataset(directory, img_size, batch_size, num_classes, augment=False, shuffle=True):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(directory))
    print(class_names)
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            image_paths.append(fpath)
            labels.append(class_indices[class_name])

    # Convert lists to tensors
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels)

    # Check for invalid labels
    valid_indices = tf.where((labels >= 0) & (labels < num_classes))
    image_paths = tf.gather(image_paths, valid_indices)
    labels = tf.gather(labels, valid_indices)

    # Convert tensors to scalar strings
    image_paths = tf.squeeze(image_paths, axis=1)
    labels = tf.squeeze(labels, axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

def get_data_generators(train_dir, test_dir, img_size, batch_size, num_classes):
    train_dataset = get_dataset(train_dir, img_size, batch_size, num_classes, augment=True)
    validation_dataset = get_dataset(test_dir, img_size, batch_size, num_classes, shuffle=False)
    return train_dataset, validation_dataset
