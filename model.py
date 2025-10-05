import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Dropout
from tensorflow.keras.utils import plot_model

def create_dual_cnn_model(input_shape=(32, 32, 3), num_classes=2)
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        model: Compiled Keras model
    """
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Branch 1: Smaller filters (3x3) 
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='branch1_conv1')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='branch1_conv2')(branch1)
    branch1 = MaxPooling2D((2, 2), name='branch1_pool1')(branch1)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='branch1_conv3')(branch1)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='branch1_conv4')(branch1)
    branch1 = MaxPooling2D((2, 2), name='branch1_pool2')(branch1)
    
    # Branch 2: Larger filters (5x5, 7x7) 
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same', name='branch2_conv1')(inputs)
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same', name='branch2_conv2')(branch2)
    branch2 = MaxPooling2D((2, 2), name='branch2_pool1')(branch2)
    branch2 = Conv2D(64, (7, 7), activation='relu', padding='same', name='branch2_conv3')(branch2)
    branch2 = Conv2D(64, (7, 7), activation='relu', padding='same', name='branch2_conv4')(branch2)
    branch2 = MaxPooling2D((2, 2), name='branch2_pool2')(branch2)
    
    # Concatenate both branches
    concatenated = Concatenate(name='concatenate_branches')([branch1, branch2])
    
    # Additional layers after concatenation
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='post_concat_conv')(concatenated)
    x = MaxPooling2D((2, 2), name='post_concat_pool')(x)
    x = Dropout(0.5, name='dropout1')(x)
    
    # Flatten and dense layers
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(128, activation='relu', name='dense2')(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='Dual_CNN_Model')
    
    return model

# Alternative simpler version with different filter sizes in parallel
def create_simple_dual_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Simpler version with parallel convolution layers using different filter sizes
    """
    
    inputs = Input(shape=input_shape)
    
    # Parallel convolution layers with different filter sizes
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    conv7x7 = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    
    # Concatenate all parallel convolutions
    concatenated = Concatenate()([conv3x3, conv5x5, conv7x7])
    
    # Continue with common layers
    x = MaxPooling2D((2, 2))(concatenated)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Simple_Dual_CNN')
    
    return model

# Create and compile the model
def main():
    # Create the model
    model = create_dual_cnn_model(input_shape=(32, 32, 3), num_classes=10)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    model.summary()
    
    try:
        plot_model(model, 'dual_cnn_model.png', show_shapes=True)
        print("Model architecture saved as 'dual_cnn_model.png'")
    except:
        print("Could not plot model. Make sure graphviz and pydot are installed.")
    
    return model

def train_with_data():
   
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.load_data()
    
    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create model
    model = create_dual_cnn_model(input_shape=(32, 32, 3), num_classes=10)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    model = main()
    
