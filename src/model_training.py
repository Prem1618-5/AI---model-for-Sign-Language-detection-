"""
Model Training Module for Sign Language Detection ML Project

This module handles the training of the machine learning model for
gesture recognition using preprocessed hand landmark data.
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data_preprocessing import GestureDataProcessor

class GestureModelTrainer:
    """
    Handles the training of ML models for gesture recognition.
    
    Attributes:
        model_dir (str): Directory to save trained models
        model (tf.keras.Model): The TensorFlow model
        history (tf.keras.callbacks.History): Training history
        class_names (list): List of gesture class names
        input_shape (tuple): Shape of input features
        random_seed (int): Random seed for reproducibility
        is_two_handed (bool): Flag to track if model handles two-handed data
    """
    
    def __init__(self, model_dir='../models', random_seed=42):
        """
        Initialize the GestureModelTrainer with specified parameters.
        
        Args:
            model_dir (str): Directory to save trained models
            random_seed (int): Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.model = None
        self.history = None
        self.class_names = []
        self.input_shape = None
        self.random_seed = random_seed
        self.is_two_handed = False  # Flag to track if model handles two-handed data
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
    
    def build_model(self, input_shape, num_classes, is_two_handed=False):
        """
        Build a neural network model for gesture recognition.
        
        Args:
            input_shape (tuple): Shape of input features
            num_classes (int): Number of gesture classes
            is_two_handed (bool): Whether the model handles two-handed gestures
            
        Returns:
            tf.keras.Model: Compiled model
        """
        self.input_shape = input_shape
        self.is_two_handed = is_two_handed
        
        # Create a sequential model
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First hidden layer
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def build_lstm_model(self, input_shape, num_classes):
        """
        Build an LSTM-based model for sequential gesture recognition.
        
        Args:
            input_shape (tuple): Shape of input features
            num_classes (int): Number of gesture classes
            
        Returns:
            tf.keras.Model: Compiled model
        """
        self.input_shape = input_shape
        
        # For LSTM, we reshape the input to (1, input_features)
        # This simulates a sequence of length 1 with input_features dimensions
        lstm_input_shape = (1, input_shape[0])
        
        # Create model
        model = models.Sequential([
            # Reshape layer
            layers.Reshape(lstm_input_shape, input_shape=input_shape),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, class_names,
             epochs=50, batch_size=32, model_type='dense', is_two_handed=False):
        """
        Train the model on the provided data.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            class_names (list): List of class names
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            model_type (str): Type of model to train ('dense' or 'lstm')
            is_two_handed (bool): Whether the model handles two-handed gestures
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        self.class_names = class_names
        num_classes = len(class_names)
        self.is_two_handed = is_two_handed
        
        # Build the model based on type
        if model_type.lower() == 'lstm':
            self.build_lstm_model((X_train.shape[1],), num_classes)
        else:
            self.build_model((X_train.shape[1],), num_classes, is_two_handed)
        
        # Set up callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train the model
        print(f"Starting model training ({model_type} model)...")
        if is_two_handed:
            print("Using two-handed gesture data")
            
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        self.save_model(model_type)
        
        return self.history
    
    def save_model(self, model_type):
        """
        Save the trained model and related information.
        
        Args:
            model_type (str): Type of the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"gesture_recognition_{model_type}_model")
        self.model.save(model_path)
        
        # Save model metadata
        metadata = {
            'model_type': model_type,
            'class_names': self.class_names,
            'input_shape': self.input_shape[0] if self.input_shape else None,
            'num_classes': len(self.class_names),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'is_two_handed': self.is_two_handed
        }
        
        with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model and metadata saved to {self.model_dir}")
    
    def load_model(self, model_path=None):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            tf.keras.Model: Loaded model
        """
        if model_path is None:
            # Try to find the latest model in the model directory
            model_files = [f for f in os.listdir(self.model_dir) 
                          if os.path.isdir(os.path.join(self.model_dir, f)) and 
                          f.startswith("gesture_recognition_")]
            
            if not model_files:
                raise FileNotFoundError(f"No model files found in {self.model_dir}")
            
            # Get the most recent model file
            model_path = os.path.join(self.model_dir, sorted(model_files)[-1])
        
        # Load model
        self.model = models.load_model(model_path)
        
        # Load metadata if available
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.class_names = metadata.get('class_names', [])
            self.input_shape = (metadata.get('input_shape'),)
            self.is_two_handed = metadata.get('is_two_handed', False)
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Train or load a model first.")
        
        # Evaluate the model
        print("Evaluating model on test data...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        
        # Print evaluation results
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot training history if available
        if self.history:
            self.plot_training_history()
        
        # Return evaluation metrics
        metrics = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix for model evaluation.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the plot
        cm_path = os.path.join(self.model_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
    
    def plot_training_history(self):
        """
        Plot training history (accuracy and loss curves).
        """
        if not self.history:
            print("No training history available.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        history_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(history_path)
        plt.close()
        
        print(f"Training history plot saved to {history_path}")
    
    def predict(self, landmarks):
        """
        Make a prediction for a single set of landmarks.
        
        Args:
            landmarks (list or numpy.ndarray): Hand landmark features
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if self.model is None:
            raise ValueError("No model for prediction. Train or load a model first.")
        
        # Ensure landmarks are the right shape
        if isinstance(landmarks, list):
            features = np.array(landmarks)
        else:
            features = landmarks
        
        # Reshape for model input if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get the top prediction
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        # Get class name
        if self.class_names and predicted_class_idx < len(self.class_names):
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = str(predicted_class_idx)
        
        return predicted_class, float(confidence)

if __name__ == "__main__":
    # Example usage
    # Load processed data
    data_processor = GestureDataProcessor()
    
    try:
        # Try to load processed data
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = data_processor.load_processed_data()
        print("Loaded processed data successfully.")
    except FileNotFoundError:
        # Process raw data if processed data doesn't exist
        print("No processed data found. Processing raw data...")
        X_train, y_train, X_val, y_val, X_test, y_test, class_names = data_processor.prepare_dataset()
    
    # Initialize model trainer
    trainer = GestureModelTrainer()
    
    # Train the model
    history = trainer.train(
        X_train, y_train, X_val, y_val, class_names,
        epochs=50, batch_size=32, model_type='dense'
    )
    
    # Evaluate the model
    evaluation = trainer.evaluate(X_test, y_test)
    
    print("\nTraining and evaluation complete!") 