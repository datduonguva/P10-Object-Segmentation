import os
import tensorflow as tf
def convert_to_tf(log_dir):
    """
    This functions is for converting an
    end-to-end tensorflow model to tensorflow lite model
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(log_dir, "serve"))
    tflite_model = converter.convert()
    with open(os.path.join(log_dir, "best_model.tflite"), "wb") as f:
        f.write(tflite_model)


if __name__ == '__main__':
    log_dir = r"/home/datduong/gdrive/projects/P10-Object-Segmentation/logs/004"
    convert_to_tf(log_dir)
