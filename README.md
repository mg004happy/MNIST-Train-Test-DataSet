# 🧠 Neural Network from Scratch - Fashion MNIST Classifier

This repository contains a full implementation of a deep neural network from scratch using **NumPy**, trained on the **Fashion-MNIST** dataset. The goal is to classify clothing images into one of 10 categories without using high-level deep learning libraries like TensorFlow or PyTorch.

---

## 📁 Dataset

- The dataset is based on **Fashion-MNIST**, which is a drop-in replacement for MNIST but contains fashion product images.
- Each image is 28x28 grayscale pixels.
- There are 10 classes:
  - 0: T-shirt/top
  - 1: Trouser
  - 2: Pullover
  - 3: Dress
  - 4: Coat
  - 5: Sandal
  - 6: Shirt
  - 7: Sneaker
  - 8: Bag
  - 9: Ankle boot

---

## 🧩 Features Included

### ✅ Data Preprocessing
- Normalization: Pixel values are divided by 255.0 to scale them between 0 and 1.
- Standardization: Each feature is zero-centered and scaled to unit variance.
- One-hot encoding: Labels are converted from scalar integers to binary class matrices using `OneHotEncoder`.

---

### ✅ Activation Functions
- **ReLU (Rectified Linear Unit)** and its derivative
- **Sigmoid** and its derivative
- **Softmax** for multi-class classification in the output layer

---

### ✅ Network Architecture
The architecture is flexible and defined by:
```python
layers = [784, 512, 256, 128, 10]
activation = ['ReLU', 'ReLU', 'ReLU', 'softmax']
