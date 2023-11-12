from prepare_data import load_images_and_labels
from tested_models.tested_models import get_conv2d_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime as dt

images, labels = load_images_and_labels()

model = get_conv2d_model()

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'model_checkpoints/xg_conv2d_mae{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.hdf5',
                                                      save_best_only=True)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(x_train, y_train, epochs=50, callbacks=[model_checkpoint], validation_data=(x_test, y_test))
