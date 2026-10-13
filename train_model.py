import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load and prepare images with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_data = train_datagen.flow_from_directory(
    'dataset/', 
    target_size=(224, 224), 
    batch_size=16, 
    class_mode='binary', 
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'dataset/', 
    target_size=(224, 224), 
    batch_size=16, 
    class_mode='binary', 
    subset='validation'
)

# 2. Load VGG16 pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 3. Build Transfer Learning Architecture
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs, name="vgg16_pan_detector")

# 4. Compile and Train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("Starting VGG16 Transfer Learning Training...")
model.fit(train_data, validation_data=val_data, epochs=10)

# 5. Save the upgraded model
model.save('pan_detector.keras')
print("Training complete! Model saved as pan_detector.keras")