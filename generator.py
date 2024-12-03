from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Tanh

def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        BatchNormalization(),
        ReLU(),
        Dense(256),
        BatchNormalization(),
        ReLU(),
        Dense(output_dim, activation='tanh')
    ])
    return model
