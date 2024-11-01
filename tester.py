import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Loading the model
model = tf.keras.models.load_model('M4_v3.1.1.keras')

# Class names order should be the same as used during training the model
class_names = ['african_hunting_dog', 'ant', 'ashcan', 'black_footed_ferret', 'bookshop', 'carousel', 
               'catamaran', 'cocktail_shaker', 'combination_lock', 'consomme', 'coral_reef', 'dalmatian', 
               'dishrag', 'fire_screen', 'goose', 'green_mamba', 'king_crab', 'ladybug', 'lion', 'lipstick', 
               'miniature_poodle', 'orange', 'organ', 'parallel_bars', 'photocopier', 'rhinoceros_beetle', 
               'slot', 'snorkel', 'spider_web', 'toucan', 'triceratops', 'unicycle', 'vase']

img_height, img_width = 84, 84  
occlusion_size = (28, 28)  # Size of each occlusion block for 84x84 image

# Positions for the 9 blocks in the 3x3 grid
positions = [
    (0, 0), (28, 0), (56, 0),  # First row (Top)
    (0, 28), (28, 28), (56, 28),  # Second row (Middle)
    (0, 56), (28, 56), (56, 56)   # Third row (Bottom)
]

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize((img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return img_array

def apply_occlusion(img_array, position, occlusion_size):
    occluded_img = img_array.copy()
    start_x, start_y = position
    occluded_img[start_y:start_y+occlusion_size[1], start_x:start_x+occlusion_size[0], :] = 0
    return occluded_img

def predict_image_class(image_path, occlusion_position=None):
    img_array = load_and_preprocess_image(image_path)
    if occlusion_position:
        img_array = apply_occlusion(img_array, occlusion_position, occlusion_size)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])    
    class_id = np.argmax(score)
    return class_names[class_id], 100 * np.max(score), img_array[0].numpy()

def browse_image():
    for widget in grid_frame.winfo_children():
        widget.destroy()

    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((1200, 1200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        for idx, position in enumerate(positions):
            class_name, confidence, occluded_image = predict_image_class(file_path, position)
            occluded_image = Image.fromarray((occluded_image * 255).astype('uint8'))
            occluded_image = ImageTk.PhotoImage(occluded_image)
            img_label = tk.Label(grid_frame, image=occluded_image)
            img_label.image = occluded_image
            img_label.grid(row=2*(idx // 3), column=idx % 3)
            text_label = tk.Label(grid_frame, text=f"{class_name}\n{confidence:.2f}%", pady=10)
            text_label.grid(row=2*(idx // 3) + 1, column=idx % 3)

# Create the GUI for testing
root = tk.Tk()
root.title("Image Classifier with Occlusion")

frame = tk.Frame(root)
frame.pack(pady=20)

btn = tk.Button(frame, text="Browse Image", command=browse_image)
btn.pack()

panel = tk.Label(frame)
panel.pack()

grid_frame = tk.Frame(root)
grid_frame.pack(pady=20)

root.mainloop()
