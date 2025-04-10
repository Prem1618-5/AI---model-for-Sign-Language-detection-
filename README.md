# Sign Language Detection System using Machine Learning

A system that detects and interprets sign language gestures using machine learning techniques.

## Overview

This project implements a complete pipeline for sign language detection and recognition:

1. **Data Collection**: Capture custom hand gesture data using webcam and MediaPipe hand tracking (supports both single and two-handed gestures)
2. **Data Preprocessing**: Normalize and augment hand landmark data for training
3. **Model Training**: Train neural network models to recognize hand gestures
4. **Real-time Recognition**: Perform real-time gesture recognition and sequence detection

## Project Structure

```
sign_language_ml/
├── data/                  # Data storage
│   ├── raw/               # Raw gesture data (JSON files)
│   └── processed/         # Processed datasets (NPZ files)
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks (for exploration and visualization)
├── src/                   # Source code
│   ├── data_collection.py # Gesture data collection module
│   ├── data_preprocessing.py # Data preprocessing module
│   ├── model_training.py  # Model training module
│   ├── realtime_recognition.py # Real-time recognition module
│   └── main.py            # Main CLI entry point
└── requirements.txt       # Python dependencies
```

## Detailed Module Explanation

### `data_collection.py`

This module handles collecting hand gesture data using your webcam and MediaPipe's hand tracking technology.

**Key Functions and Features**:

- **`DataCollector` Class**: 
  - **Purpose**: Main class for collecting gesture data using webcam input
  - **Key Methods**:
    - `__init__(output_dir, capture_delay, min_detection_confidence, min_tracking_confidence)`: 
      - **Purpose**: Initializes the data collector with specified parameters
      - **Parameters**:
        - `output_dir`: Directory where collected data will be saved
        - `capture_delay`: Time delay between frame captures (seconds)
        - `min_detection_confidence`: Minimum confidence required for hand detection
        - `min_tracking_confidence`: Minimum confidence required for hand tracking
    
    - `collect_gesture_data(gesture_name, num_samples)`: 
      - **Purpose**: Captures video frames of hand gestures and saves landmark data
      - **Parameters**:
        - `gesture_name`: Name of the gesture being collected
        - `num_samples`: Number of samples to collect
      - **Process**:
        - Opens webcam in a resizable window
        - Detects and tracks hands in the video feed
        - Captures hand landmark data when space key is pressed
        - Saves data in JSON format with both hands' landmarks (when available)
        - Includes error handling to prevent crashes during data collection
    
    - `collect_multiple_gestures(gestures_config)`:
      - **Purpose**: Collects data for multiple gestures in sequence
      - **Parameters**:
        - `gestures_config`: List of dictionaries with gesture names and sample counts
      - **Process**:
        - Loops through each gesture in the config
        - Calls `collect_gesture_data` for each one
        - Returns a list of saved file paths

- **`extract_landmarks_from_file(filepath)`**:
  - **Purpose**: Utility function to extract landmark data from saved JSON files
  - **Parameters**:
    - `filepath`: Path to the JSON file containing gesture data
  - **Returns**: Tuple of (gesture_name, landmarks_array)

**Two-handed Gesture Support**:
- The module can detect and store data for both hands simultaneously
- The data format includes a list of hands for each frame, with each hand containing 21 landmarks
- This enables recognition of gestures that require both hands, like "applause", "thank you", etc.

### `data_preprocessing.py`

This module handles the preprocessing of raw gesture data into a format suitable for training machine learning models.

**Key Functions and Features**:

- **`GestureDataProcessor` Class**:
  - **Purpose**: Processes raw gesture data for model training
  - **Key Methods**:
    - `__init__(data_dir, processed_dir, random_seed)`:
      - **Purpose**: Initializes the data processor
      - **Parameters**:
        - `data_dir`: Directory containing raw gesture data files
        - `processed_dir`: Directory to save processed data
        - `random_seed`: Random seed for reproducibility
    
    - `load_gesture_data(file_pattern)`:
      - **Purpose**: Loads all gesture data files from the data directory
      - **Parameters**:
        - `file_pattern`: Pattern to match gesture data files (e.g., "*.json")
      - **Returns**: Dictionary with gesture names as keys and lists of landmark samples, plus a flag indicating if the data contains two-handed gestures
    
    - `normalize_landmarks(landmarks)`:
      - **Purpose**: Normalizes hand landmarks to be scale and translation-invariant
      - **Process**:
        - Centers the hand at the palm center
        - Scales the hand to a standard size
        - Makes the gesture recognition work regardless of hand position/size
    
    - `flatten_landmarks(landmarks)`:
      - **Purpose**: Converts 3D landmarks to a flat feature vector for ML models
    
    - `augment_landmarks(landmarks, num_augmentations)`:
      - **Purpose**: Generates augmented data with rotations, translations, and scaling
      - **Parameters**:
        - `landmarks`: Landmark data to augment
        - `num_augmentations`: Number of augmented versions to generate
      - **Process**:
        - Applies small random rotations
        - Applies small random translations
        - Applies small random scaling
        - Helps increase dataset size and model robustness
    
    - `prepare_dataset(augment, test_size, val_size)`:
      - **Purpose**: Prepares the complete dataset for training
      - **Parameters**:
        - `augment`: Whether to perform data augmentation
        - `test_size`: Proportion of data for testing
        - `val_size`: Proportion of data for validation
      - **Process**:
        - Loads and normalizes gesture data
        - Handles both single-handed and two-handed gesture formats
        - Performs data augmentation if requested
        - Splits data into training, validation, and test sets
    
    - `save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, class_names, metadata)`:
      - **Purpose**: Saves processed data to disk in NPZ format
    
    - `load_processed_data()`:
      - **Purpose**: Loads previously processed data
      - **Returns**: Train/val/test splits, class names, and two-handed flag

**Two-handed Data Processing**:
- Detects and processes two-handed gesture data
- For two-handed gestures, processes each hand separately and then combines the features
- The resulting feature vector is twice the size of single-handed gestures (126 vs 63 dimensions)

### `model_training.py`

This module handles training and evaluating machine learning models for gesture recognition.

**Key Functions and Features**:

- **`GestureModelTrainer` Class**:
  - **Purpose**: Trains and evaluates gesture recognition models
  - **Key Methods**:
    - `__init__(model_dir, random_seed)`:
      - **Purpose**: Initialize the model trainer
      - **Parameters**:
        - `model_dir`: Directory to save trained models
        - `random_seed`: Random seed for reproducibility
    
    - `build_model(input_shape, num_classes, is_two_handed)`:
      - **Purpose**: Creates a dense neural network for gesture recognition
      - **Parameters**:
        - `input_shape`: Shape of input features
        - `num_classes`: Number of gesture classes
        - `is_two_handed`: Whether the model handles two-handed gestures
      - **Architecture**:
        - Input layer matching the feature dimensions (63 for single-hand, 126 for two-handed)
        - Two hidden dense layers with ReLU activation
        - Batch normalization and dropout for regularization
        - Softmax output layer for classification
    
    - `build_lstm_model(input_shape, num_classes)`:
      - **Purpose**: Creates an LSTM-based model for sequential gesture recognition
      - **Architecture**:
        - Reshapes input to treat features as a sequence
        - Two LSTM layers for temporal processing
        - Dense layers for final classification
    
    - `train(X_train, y_train, X_val, y_val, class_names, epochs, batch_size, model_type, is_two_handed)`:
      - **Purpose**: Trains the model with provided data
      - **Parameters**:
        - `X_train, y_train`: Training data and labels
        - `X_val, y_val`: Validation data and labels
        - `class_names`: Names of the gesture classes
        - `epochs`: Number of training epochs
        - `batch_size`: Batch size for training
        - `model_type`: Type of model ('dense' or 'lstm')
        - `is_two_handed`: Flag for two-handed gesture processing
      - **Features**:
        - Uses early stopping to prevent overfitting
        - Uses learning rate reduction on plateau
        - Saves training history for visualization
    
    - `save_model(model_type)`:
      - **Purpose**: Saves the trained model and metadata
      - **Saved Information**:
        - Model weights
        - Class names
        - Input shape
        - Two-handed flag
    
    - `load_model(model_path)`:
      - **Purpose**: Loads a previously trained model
    
    - `evaluate(X_test, y_test)`:
      - **Purpose**: Evaluates model performance on test data
      - **Metrics**:
        - Accuracy
        - Loss
        - Detailed classification report
        - Confusion matrix visualization
    
    - `plot_confusion_matrix(y_true, y_pred)`:
      - **Purpose**: Visualizes model predictions with a confusion matrix
    
    - `plot_training_history()`:
      - **Purpose**: Plots accuracy and loss curves from training
    
    - `predict(landmarks)`:
      - **Purpose**: Makes predictions on new gesture data
      - **Returns**: Predicted gesture and confidence score

**Two-handed Model Support**:
- Models can be trained to recognize gestures using data from both hands
- The input layer automatically adapts to the larger feature dimensions
- Metadata tracks whether the model is trained for two-handed gestures

### `realtime_recognition.py`

This module provides real-time sign language recognition using a webcam and trained model.

**Key Functions and Features**:

- **`RealtimeGestureRecognizer` Class**:
  - **Purpose**: Performs real-time gesture recognition from webcam feed
  - **Key Methods**:
    - `__init__(model_path, min_detection_confidence, min_tracking_confidence, recognition_threshold, smoothing_window)`:
      - **Purpose**: Initializes the recognizer
      - **Parameters**:
        - `model_path`: Path to the trained model
        - `min_detection_confidence`: Minimum confidence for hand detection
        - `min_tracking_confidence`: Minimum confidence for hand tracking
        - `recognition_threshold`: Minimum confidence for gesture recognition
        - `smoothing_window`: Window size for temporal smoothing
    
    - `preprocess_landmarks(landmarks, is_left_hand)`:
      - **Purpose**: Processes landmarks from the webcam feed for model input
      - **Parameters**:
        - `landmarks`: Hand landmarks from MediaPipe
        - `is_left_hand`: Whether the hand is the left hand
    
    - `preprocess_two_hands(left_hand_landmarks, right_hand_landmarks)`:
      - **Purpose**: Processes landmarks from both hands for two-handed gestures
      - **Parameters**:
        - `left_hand_landmarks`: Landmarks for the left hand
        - `right_hand_landmarks`: Landmarks for the right hand
      - **Process**:
        - Normalizes and processes each hand separately
        - Combines features from both hands for model input
    
    - `get_smoothed_prediction()`:
      - **Purpose**: Applies temporal smoothing to predictions to reduce jitter
      - **Process**:
        - Maintains a history buffer of recent predictions
        - Returns the most frequent gesture if it appears in >60% of frames
    
    - `update_sequence(gesture, confidence)`:
      - **Purpose**: Maintains a sequence of recognized gestures
      - **Process**:
        - Adds new gestures to the sequence
        - Handles timeouts between gestures
    
    - `get_sequence_text()`:
      - **Purpose**: Returns the current gesture sequence as text
    
    - `run(camera_id, flip_image)`:
      - **Purpose**: Main method that runs the recognition loop
      - **Parameters**:
        - `camera_id`: Camera device ID
        - `flip_image`: Whether to flip the camera image horizontally
      - **Process**:
        - Opens webcam in a resizable window
        - Detects hand landmarks using MediaPipe
        - Identifies left and right hands when both are present
        - Processes the landmarks for prediction
        - Applies temporal smoothing
        - Displays recognition results in real-time
        - Maintains and displays gesture sequences

**Two-handed Recognition Features**:
- Detects and labels both left and right hands
- Can process gestures that require both hands
- Shows hand labels on the video feed
- Combines features from both hands for prediction

### `main.py`

This module provides a command-line interface to run different components of the system.

**Key Functions and Features**:

- **Command-line Interface**:
  - **Purpose**: Provides a unified interface for the entire pipeline
  - **Commands**:
    - `collect`: Command to collect gesture data
      - **Options**:
        - `--gestures`: Names of gestures to collect
        - `--samples`: Number of samples per gesture
        - `--output`: Output directory
    
    - `preprocess`: Command to preprocess collected data
      - **Options**:
        - `--augment`: Enable data augmentation
        - `--input`: Input directory with raw data
        - `--output`: Output directory for processed data
    
    - `train`: Command to train a recognition model
      - **Options**:
        - `--model-type`: Type of model ('dense' or 'lstm')
        - `--epochs`: Number of training epochs
        - `--batch-size`: Training batch size
        - `--data`: Directory with processed data
        - `--output`: Output directory for the model
    
    - `evaluate`: Command to evaluate a trained model
      - **Options**:
        - `--model`: Path to the trained model
        - `--data`: Directory with processed data
    
    - `recognize`: Command to run real-time recognition
      - **Options**:
        - `--model`: Path to the trained model
        - `--camera`: Camera device ID
        - `--threshold`: Recognition confidence threshold
        - `--no-flip`: Disable horizontal flipping of camera image

## Usage Instructions

### Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Collection

To collect data for custom gestures:

```bash
python src/main.py collect --gestures hello thank_you yes no --samples 50
```

This will:
- Open your webcam in a resizable window
- Show real-time hand tracking
- Capture 50 samples each for the gestures "hello", "thank_you", "yes", and "no"
- Store the landmark data in JSON files in the `data/raw` directory

**Tips for Data Collection**:
- Make sure there's good lighting
- Position your hands within the camera frame
- For two-handed gestures, make sure both hands are visible
- Press SPACE to start collecting after positioning your hands
- Press Q to quit early

### Data Preprocessing

To preprocess the collected data:

```bash
python src/main.py preprocess --augment
```

This will:
- Load all raw gesture data
- Normalize and process the landmarks
- Augment the data to increase the dataset size (if --augment is used)
- Split the data into training, validation, and test sets
- Save the processed data to the `data/processed` directory

### Model Training

To train a model with the preprocessed data:

```bash
python src/main.py train --model-type dense --epochs 50
```

This will:
- Load the preprocessed data
- Build a neural network model (dense or LSTM)
- Train the model on the training data
- Validate on the validation set
- Evaluate on the test set
- Save the trained model to the `models` directory
- Generate performance visualizations

### Real-time Recognition

To run real-time recognition with a webcam:

```bash
python src/main.py recognize --threshold 0.7
```

This will:
- Load the trained model
- Open your webcam in a resizable window
- Track hand movements in real-time
- Recognize gestures when confidence exceeds the threshold
- Display the recognized gestures and confidence scores
- Build a sequence of gestures over time
- Allow window resizing with the 'r' key

## Technical Details

### Hand Landmark Detection

The system uses MediaPipe Hands to detect 21 landmarks on each hand:
- Wrist point (1 landmark)
- Thumb (4 landmarks)
- Index finger (4 landmarks)
- Middle finger (4 landmarks)
- Ring finger (4 landmarks)
- Pinky finger (4 landmarks)

Each landmark has x, y, z coordinates normalized to the image dimensions.

### Two-handed Gesture Processing

The system can handle gestures that require both hands:
1. During data collection, landmarks for both hands are captured when available
2. During preprocessing, features from both hands are combined
3. During training, the model learns from the combined feature space
4. During recognition, both hands are tracked and processed together

### Data Normalization

To make the model robust to different hand positions, sizes, and orientations:
- Translation normalization centers the hand at the palm center
- Scale normalization ensures uniform size regardless of distance from camera
- Rotation invariance is achieved through data augmentation

### Model Architecture

The default dense neural network includes:
- Input layer matching the flattened landmark features (63 features for one hand, 126 for two hands)
- Two hidden dense layers with ReLU activation
- Batch normalization to stabilize training
- Dropout layers for regularization
- Output softmax layer with one unit per gesture class

### Real-time Processing Pipeline

The real-time recognition process:
1. Captures webcam frames
2. Detects hand landmarks using MediaPipe
3. Identifies left and right hands when both are present
4. Normalizes landmarks for each hand
5. Makes model predictions with combined features
6. Applies temporal smoothing for stability
7. Tracks gesture sequences
8. Displays recognition results with confidence scores

## Troubleshooting

### Camera Issues

If the webcam doesn't open or crashes:
- Check if another application is using the camera
- Try a different camera ID: `python src/main.py recognize --camera 1`
- Ensure camera permissions are granted to the application

### Recognition Problems

If gesture recognition is poor:
- Collect more samples of problematic gestures
- Try with better lighting conditions
- Ensure both hands are fully visible for two-handed gestures
- Lower the recognition threshold: `python src/main.py recognize --threshold 0.5`

### Training Issues

If model training fails or performs poorly:
- Check if you have enough samples for each gesture
- Try a longer training duration with more epochs
- Use data augmentation to increase dataset size
- Consider using a different model architecture
