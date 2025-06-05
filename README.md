# Image Caption Generator using CNN-LSTM from Scratch

## Project Overview

This project generates descriptive captions for images using a deep learning model built from scratch. The model combines Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for caption generation. The project includes image preprocessing, feature extraction, sequence generation, model training, and evaluation.

## Technologies Used

- **CNN**: InceptionV3 (pre-trained on ImageNet) for image feature extraction
- **LSTM**: Long Short-Term Memory for sequential caption generation
- **Frameworks**: Keras, TensorFlow
- **Other Libraries**: Keras Tuner, NumPy, Pickle

## Dataset

The dataset contains 8,091 real-world images in `.jpg` format, each associated with five different human-written captions. The dataset is split as follows:
- **Training**: 70% (5,664 images)
- **Validation**: 15% (1,214 images)
- **Test**: 15% (1,214 images)

Example captions for one image:
- A child in a pink dress is climbing up a set of stairs in an entryway.
- A girl going into a wooden building.
- A little girl climbing into a wooden playhouse.
- A little girl climbing the stairs to her playhouse.
- A little girl in a pink dress goes into a wooden cabin.

## Workflow

1. **Data Preprocessing**:
   - **Captions**: Cleaned (lowercased), tokenized, padded with "startseq" and "endseq".
   - **Images**: Resized to 299x299 pixels, normalized, and processed using InceptionV3 to extract 2048-dimensional features.

2. **Feature Extraction**: 
   - The features are extracted from InceptionV3 and saved in a file `custom_image_features.pkl`.

3. **Sequence Generation**: 
   - Merged image features with corresponding captions to create triplets: `(input_image, input_sequence, output_word)`.
   - Saved these triplets as `.npy` files for efficient training.

4. **Model Architecture**:
   - **CNN Branch**: InceptionV3 for image feature extraction followed by a Dense layer for feature transformation.
   - **Caption Branch**: LSTM to process the captions with embedding and LSTM layers for generating the next word in the sequence.
   - **Fusion**: The image features are combined with the LSTM output and passed through Dense layers for the final output.

## Training Configuration

- **Loss Function**: Sparse Categorical Cross-Entropy
- **Optimizer**: Adam (Learning Rate = 0.001)
- **Batch Size**: 64
- **Epochs**: 20 (with early stopping based on validation loss)

## Evaluation

- **Test Accuracy**: 38.4%
- **Test Loss**: 3.44
- **Validation Accuracy (Cross-Validation)**: ~39.6%

Sample generated captions:
- "A dog is running through the sand."
- "A little girl in a pink shirt is jumping on a trampoline."
- "A man in a white shirt is playing a game of basketball."

## Hyperparameter Tuning and Cross-Validation

- **Hyperparameters Tuned**: Dropout rate, LSTM units, learning rate.
- **Cross-Validation**: 2-fold cross-validation on a subset of the training data to ensure generalization and prevent overfitting.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Thisakya-Anudini/Caption-Generator.git
   cd Caption-Generator
