# Linear Regression for Command Prediction

This project demonstrates a linear regression model applied to a simple command-line prediction task, designed in Python. The model predicts the next word in a command sequence based on input sequences of command-line phrases.

## Code Overview

This project explores the basic principles of machine learning with `scikit-learn` by building a linear regression model that learns simple patterns in command sequences.

### Key Steps in the Process

1. **Dataset Generation**:
   - A synthetic dataset is created using common terminal commands, like file operations (`cp`, `mv`, `rm`) and system queries (`ls`, `ps aux`, `df -h`).
   - This list is then replicated multiple times to increase the sample size.

2. **Data Preprocessing**:
   - Each command is split into words, generating input-output pairs where the input is the command sequence (without the last word), and the output is the last word.
   - The sequences are padded to a uniform length, ensuring compatibility with machine learning models that require fixed input sizes.

3. **Feature Engineering**:
   - One-hot encoding transforms text data into numerical form.
   - Separate encoders handle the input sequence and target word encoding, converting both into formats that the linear regression model can interpret.

4. **Model Training**:
   - Using `scikit-learn`'s `LinearRegression` model, the encoded dataset is split into training and testing sets.
   - The model is then trained on this dataset, learning relationships between command words.

5. **Prediction**:
   - A prediction function takes an input sequence, processes and encodes it, and then uses the model to predict the next word.
   - The predicted word is decoded from its encoded form back into text.

## Installation and Setup

To run this project, you'll need to install the required dependencies. These include libraries for data processing, machine learning, and optional visualization.

Install these dependencies with:

```bash
pip install numpy scikit-learn tensorflow aiofiles gradio seaborn --user --upgrade --break-system-packages
```

> **Note:** Make sure to install these packages in an isolated environment (such as a virtual environment or conda environment) to avoid dependency conflicts.

## Usage

The main script, `main.py`, can be executed directly to train the model, evaluate its performance, and run a simple prediction example.

1. **Run the Script**: Execute the following command:

   ```bash
   python main.py
   ```

2. **Expected Output**:
   - The script will output training and testing scores, demonstrating how well the model fits the synthetic dataset.
   - It will also display a sample prediction where it attempts to guess the next word of a provided input sequence.

### Model Limitations

This model has several known limitations:

1. **Simplistic Model**: Using linear regression for sequence prediction is limited in capturing complex patterns, making it unsuitable for real-world command prediction.
2. **Synthetic Dataset**: The dataset is small and artificially generated, which limits the model's accuracy.
3. **Limited Vocabulary**: The model cannot generalize well to commands outside of the given set due to the small and specific dataset.
4. **Padding Influence**: Sequence padding may negatively affect the prediction quality, as padding introduces non-informative tokens.

