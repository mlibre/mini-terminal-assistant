import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic terminal command dataset
def generate_terminal_commands():
    commands = [
        "ls -la /home",
        "cd /usr/local/bin",
        "cp file1.txt file2.txt",
        "mv document.pdf Downloads",
        "rm temp.txt",
        "mkdir new_directory",
        "touch newfile.txt",
        "chmod 755 script.sh",
        "grep pattern file.txt",
        "find . -name *.py",
        "ps aux | grep process",
        "df -h",
        "pwd",
        "tar -czf archive.tar.gz files",
        "wget https://example.com/file"
    ] * 10  # Multiply the list to get more samples
    return commands

# Step 2: Preprocess the data
def preprocess_commands(commands):
    # Split commands into words
    command_sequences = [cmd.split() for cmd in commands]
    
    # Create input sequences (all words except last) and targets (last word)
    X = []
    y = []
    
    for sequence in command_sequences:
        if len(sequence) > 1:  # Only use commands with at least 2 words
            # Get all words except the last one
            input_seq = sequence[:-1]
            # Pad sequences to max length
            padded_seq = input_seq + ['PAD'] * (5 - len(input_seq))
            X.append(padded_seq)
            y.append(sequence[-1])
    
    return X, y

# Step 3: Convert text to numerical features
def encode_features(X, y):
    # Initialize OneHotEncoder
    encoder_X = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_y = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Reshape X to 2D array
    X_flat = np.array(X).reshape(-1, 1)
    encoder_X.fit(X_flat)
    
    # Transform each sequence
    X_encoded = []
    for sequence in X:
        sequence_encoded = encoder_X.transform(np.array(sequence).reshape(-1, 1))
        X_encoded.append(sequence_encoded.flatten())
    
    # Encode target words
    y_encoded = encoder_y.fit_transform(np.array(y).reshape(-1, 1))
    
    return np.array(X_encoded), y_encoded, encoder_X, encoder_y

# Step 4: Create and train the model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Step 5: Function to predict next word
def predict_next_word(model, input_sequence, encoder_X, encoder_y):
    # Preprocess input sequence
    padded_seq = input_sequence + ['PAD'] * (5 - len(input_sequence))
    
    # Encode input sequence
    sequence_encoded = []
    for word in padded_seq:
        word_encoded = encoder_X.transform(np.array([word]).reshape(-1, 1))
        sequence_encoded.extend(word_encoded.flatten())
    
    # Make prediction
    prediction = model.predict([sequence_encoded])
    
    # Convert prediction back to word
    predicted_word = encoder_y.inverse_transform(prediction.reshape(1, -1))[0][0]
    
    return predicted_word

# Main execution
def main():
    # Generate dataset
    commands = generate_terminal_commands()
    
    # Preprocess data
    X, y = preprocess_commands(commands)
    
    # Encode features
    X_encoded, y_encoded, encoder_X, encoder_y = encode_features(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Calculate training and test scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("Model Performance:")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")

    # Example prediction
    test_sequence = ['ls', '-h']
    predicted = predict_next_word(model, test_sequence, encoder_X, encoder_y)
    print(f"Input: {' '.join(test_sequence)}")
    print(f"Predicted next word: {predicted}")

if __name__ == "__main__":
    main()