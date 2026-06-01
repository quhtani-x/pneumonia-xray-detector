import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# loads the trained model and checks a single chest x-ray.
# learning demo only, not a real diagnosis. run train.py first.


def main():
    if len(sys.argv) < 2:
        print("usage: python predict.py xray.jpeg")
        return

    model = tf.keras.models.load_model("pneumonia_model.keras")

    # load grayscale at the same size the model trained on
    img = image.load_img(sys.argv[1], target_size=(150, 150), color_mode="grayscale")
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)

    prob = float(model.predict(arr)[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    print(f"x-ray looks: {label} ({prob*100:.1f}% pneumonia)")
    print("note: demo only, not a medical diagnosis")


if __name__ == "__main__":
    main()
