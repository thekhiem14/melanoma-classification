import tensorflow as tf
model = tf.keras.models.load_model("alternative_model.h5")
model.save("alternative_model.keras")  # Lưu theo định dạng mới (TF >= 2.11)
