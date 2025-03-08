import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 4  # plastic, metal, biomedical, shoes
BATCH_SIZE = 32  # Increased for stability
EPOCHS = 30
DATASET_DIR = 'c:/proj1/dataset/'

# Custom multi-label generator wrapper
def convert_to_multilabel(generator):
    while True:
        images, labels = next(generator)
        # Convert one-hot to multi-label (soften labels)
        labels = np.where(labels > 0.5, 1, labels * 0.5)  # Allow partial multi-label
        yield images, labels

# Data generator with balanced augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # Reduced to avoid excessive distortion
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator_base = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=['plastic', 'metal', 'biomedical', 'shoes'],
    shuffle=True
)

validation_generator_base = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=['plastic', 'metal', 'biomedical', 'shoes'],
    shuffle=True  # Shuffle validation too
)

# Wrap generators for multi-label
train_generator = convert_to_multilabel(train_generator_base)
validation_generator = convert_to_multilabel(validation_generator_base)

# Build model with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Reduced L2
x = Dropout(0.3)(x)  # Reduced dropout
predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze fewer layers initially
for layer in base_model.layers[:-50]:  # Unfreeze more early
    layer.trainable = False

# Compile with balanced learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train initial layers
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator_base.samples // BATCH_SIZE),
    epochs=15,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator_base.samples // BATCH_SIZE),
    callbacks=[reduce_lr, early_stopping]
)

# Unfreeze more layers
for layer in base_model.layers[-70:]:  # Unfreeze more for fine-tuning
    layer.trainable = True

# Recompile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
)

# Fine-tune
history_fine = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator_base.samples // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator_base.samples // BATCH_SIZE),
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[reduce_lr, early_stopping]
)

# Save the model
model.save('c:/proj1/waste_cnn_model.h5')
print("Model saved as 'c:/proj1/waste_cnn_model.h5'")

# Plot accuracy and AUC
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['auc'] + history_fine.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'] + history_fine.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Evaluate
val_loss, val_accuracy, val_auc = model.evaluate(validation_generator_base)  # Use base for evaluation
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation AUC: {val_auc}")