import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from tensorflow.keras import backend as K
from keras.layers import LSTM
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import skfuzzy as fuzz
from scipy.stats import multivariate_normal 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrixf
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

# Load and display each .nii file
loaded_images = []
for i, image_file in enumerate(image_files):
image_path = os.path.join(image_dir, image_file)
try:
# Load the NIfTI file
img = nib.load(image_path)
img_data = img.get_fdata() # Get the image data as a NumPy array
# Print the shape of the image data
print(f"Image {i+1}: {image_file}, shape: {img_data.shape}")
# Display the middle slice of the 3D image along the first axis
middle_slice = img_data[img_data.shape[0] // 2, :, :]
plt.figure()
plt.imshow(middle_slice.T, cmap="gray", origin="lower")
plt.title(f"Image {i+1}: {image_file} - Middle Slice")
plt.axis("off")
plt.show()
except Exception as e:
print(f"Error loading image {image_file}: {e}")

loaded_images = []
for i, image_file in enumerate(image_files):
image_path = os.path.join(image_dir, image_file)
try:
# Load the NIfTI file
img = nib.load(image_path)
img_data = img.get_fdata()

flattened_images = [img.flatten() for img in loaded_images]

n_clusters = 2
m = 2

cntr, u, u0,d,jm,p, fpc = fuzz.cluster.cmeans(
    flattened_images.T, n_clusters, m, error=0.005, maxiter=1000, init=None
)

cluster = np.argmax(u, axis=0)
for cluster in range(n_clusters):
cluster_size = np.sum(cluster_labels == cluster)
print(f"Cluster {cluster} contains {cluster_size} images.")

for cluster in range(n_cluster):
cluster_size = np.sum(cluster_labels == cluster)
if cluster_size > 0:
plt.figure()
cluster_images = np.array(loaded_images)[cluster_labels == cluster]
plt.imshow(cluster_images[0][..., cluster_images[0].shape[2] // 2], cmap="gray") # Middle slice
plt.title(f"Cluster {cluster} - Sample Image")
plt.axis("off")
plt.show()
else:
print(f"Cluster {cluster} is empty.")

lstm_output_size = 70
cnn = Sequential()
generated_labels = tf.keras.utils.to_binary(generated_labels, num_classes=2)
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(256, 256,3)))
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu"))
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Convolution1D(128, 3, border_mode="same",activation="relu"))
cnn.add(Convolution1D(128, 3, border_mode="same",activation="relu"))
cnn.add(Convolution1D(128, 3, border_mode="same",activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Convolution1D(256, 3, border_mode="same",activation="relu"))
cnn.add(Convolution1D(256, 3, border_mode="same",activation="relu"))
cnn.add(Convolution1D(256, 3, border_mode="same",activation="relu"))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="sigmoid"))
cnn.fit(generated_images, generated_labels, epochs=60, batch_size=32)
cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnn-lstm/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')


