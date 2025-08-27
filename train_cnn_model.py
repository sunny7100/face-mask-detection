import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

class MaskDetectionTrainer:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = self._build_cnn_model()
        
    def _build_cnn_model(self):
        """Build CNN architecture for mask detection"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fully Connected Layers
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')  # Binary classification: Mask/No Mask
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data_generators(self, data_dir, batch_size=32):
        """Prepare data generators with augmentation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def train_model(self, data_dir, epochs=50, batch_size=32):
        """Train the CNN model"""
        
        # Prepare data
        train_gen, val_gen = self.prepare_data_generators(data_dir, batch_size)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.0001)
        ]
        
        # Train model
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, test_dir):
        """Evaluate model performance"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def main():
    """Main training function"""
    
    # Initialize trainer
    trainer = MaskDetectionTrainer()
    
    # Print model summary
    print("CNN Model Architecture:")
    trainer.model.summary()
    
    # Data directory structure should be:
    # data/
    # ├── with_mask/
    # └── without_mask/
    
    data_directory = "data"
    
    if os.path.exists(data_directory):
        print("Starting training...")
        
        # Train model
        history = trainer.train_model(
            data_dir=data_directory,
            epochs=50,
            batch_size=32
        )
        
        # Plot training history
        trainer.plot_training_history(history)
        
        # Save model
        trainer.save_model("models/mask_detection_cnn.h5")
        
        print("Training completed!")
        
    else:
        print(f"Data directory '{data_directory}' not found.")
        print("Please organize your data as:")
        print("data/")
        print("├── with_mask/")
        print("└── without_mask/")

if __name__ == "__main__":
    main()