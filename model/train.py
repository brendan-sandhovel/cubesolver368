import os
import numpy as np
from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

# Path to training dataset
filePathBase = 'data/trainingSets/'
fileExt = '.npy'


# Function to load training data
def loadData(fileNum):
    filePath = filePathBase + str(fileNum) + fileExt
    if not os.path.exists(filePath):
        print(f"Dataset file {filePath} not found!")
        return None, None  # Return None if the file doesn't exist
    
    # Load the dataset if it exists
    data = np.load(filePath).astype("float32")
    return data

# Function to initialize a new model
def initializeModel():
    print("Initializing a new model...")
    model = Sequential()
    model.add(Dense(128, input_dim=54, activation='relu'))  # Example layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))  # 12 possible moves for the Rubik's Cube
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def flatten_cube_state(cube_stickers):
    # Flatten the stickers into a 1D array of size 54 (assuming cube_stickers is a 6x3x3 matrix)
    return cube_stickers.flatten()

# Function to load a pre-trained model
def loadPretrainedModel():
    model_path = 'data/trained_model.h5'
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = keras.models.load_model(model_path)
        return model
    else:
        print("No pre-trained model found. Initializing a new model.")
        return initializeModel()

# Function to train the model (if dataset is available)
def trainModel(loadPrev=True):
    if loadPrev:
        print("Loading previous checkpoint or pre-trained model...")
        # Load pre-trained model logic here (if you have one)
        model = loadPretrainedModel()
        return model
    else:
        print("Checkpoint file does not exist. Initializing model from scratch.")
        # Initialize model from scratch
        model = initializeModel()
        
        # Check if training data exists
        i = 0
        X, Y = loadData(i)
        
        # If no data is loaded, handle it gracefully
        if X is None or Y is None:
            print("No training data found. Skipping training.")
            return model
        
        # Train the model with the available data
        print("Training from scratch.")
        model.fit(X, Y, epochs=10, batch_size=32, verbose=1)
        
        # Save the trained model for future use
        model.save('data/trained_model.h5')
        return model

# Function to get a trained model
def getTrainedModel():
    model = trainModel(loadPrev=False)  # Train the model from scratch if no checkpoint exists
    return model

# Function to make predictions with the model
def predict(stickers):
    # Check if data is available for prediction
    if stickers is None:
        print("Error: No stickers data available for prediction.")
        return None
    
    # Reshape stickers (6x3x3) to (54,) - Flatten the cube
    stickers_flat = stickers.reshape(54)
    
    # Proceed with prediction
    model = getTrainedModel()
    pred = predictMove(stickers_flat, model)
    return pred

# Helper function to predict moves (add your logic here)
def predictMove(stickers, model):
    # Flatten the stickers to match the model input shape (should be a 54-length vector)
    flattened_stickers = flatten_cube_state(stickers)
    
    # Reshape it to match model input shape
    input_data = np.expand_dims(flattened_stickers, axis=0)  # Add batch dimension
    
    # Predict the moves using the model
    predicted_moves = model.predict(input_data)
    
    # Convert prediction to move instructions (you may need to adjust this depending on the output)
    move_dict = {0: 'U', 1: 'D', 2: 'L', 3: 'R', 4: 'F', 5: 'B'}
    formatted_moves = [move_dict.get(int(move), 'X') for move in predicted_moves.flatten()]
    
    return formatted_moves  # Return a list of move instructions

if __name__ == "__main__":
    # Uncomment to generate data if needed
    #generateData(trainingSize, numFiles=numFiles)

    # Train the model, allow it to fall back to training if no valid checkpoint is found
    model = trainModel(loadPrev=True)

    # Load some data for testing or prediction
    X, Y = loadData(0)
    X = X[:20]  # Example subset
    Y = Y[:20]

    print("Input: ")
    print(X)
    print("Prediction: ")
    print(predict(X))
    print("Actual: ")
    print(Y.astype(int))
