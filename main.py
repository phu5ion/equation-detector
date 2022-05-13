import tensorflow as tf
import cv2
from utils import *
import matplotlib.pyplot as plt

DATADIR = 'equation_images'
RESIZE_H = 45 # Hardcoded to match dimensions of handwritten dataset
RESIZE_W = 45 # Hardcoded to match dimensions of handwritten dataset
PADCOLOUR = 0
CLASS_NAMES = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '*']

if __name__ == '__main__':
    # Load model
    model = tf.keras.models.load_model('eqn-detect-model')
    
    # Randomly pick an input equation image from a data folder. Please modify this to receive input from handwriting console.
    img_path = random_sample_file(DATADIR)
    input_image = cv2.imread(img_path, 0) 
    img_copy = input_image.copy()
    inverted_binary_img = binarize(input_image)
    
    # Processing image file
    bounding_box_coords = detect_contours(img_path)
    eqn_list = []
    for (x, y, w, h) in sorted(bounding_box_coords, key = lambda x: x[0]):
        img = resize_pad(inverted_binary_img[y:y+h, x:x+w], (RESIZE_H, RESIZE_W), PADCOLOUR) 
        pred_class = CLASS_NAMES[np.argmax(model.predict(tf.expand_dims(tf.expand_dims(img, 0), -1)))]
        eqn_list.append(pred_class)
    eqn = "".join(eqn_list)
    print(f"The equation is {eqn} and the answer is {eval(eqn)}")
    # for (x, y, w, h) in bounding_box_coords:
    #     cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # plt.imshow(img_copy, cmap='gray')
    # plt.show()
    
    
    
    