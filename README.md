# Pet Image Classification using Convolutional Neural Networks (CNN)

## Objective
The goal of this project is to automate the classification of pet images into **Cats** and **Dogs** for an animal welfare organization. Using TensorFlow and Keras, a Convolutional Neural Network (CNN) model is designed, trained, and evaluated to accurately identify image classes from visual features.

---

## Dataset Link
* **Dataset Name:** Dog and Cat Classification Dataset
* **Source:** Kaggle (`bhavikjikadara/dog-and-cat-classification-dataset`)
* **Access Method:** Downloaded using `kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")`

---

## Libraries Used
* `TensorFlow` / `Keras` (Model building, training, image processing)
* `scikit-learn` (Evaluation metrics and confusion matrix)
* `Matplotlib` (Visualization of graphs and sample images)
* `NumPy` (Numerical array manipulation)
* `PIL` (Image preprocessing)
* `kagglehub` (Dataset downloading)

---

## Methodology
1. **Data Understanding:** Explored the dataset structure, verified image dimensions, and extracted sample images with corresponding class labels.
2. **Preprocessing:** 
   * Resized all images to $128 \times 128$ pixels.
   * Scaled pixel values to the range $[0, 1]$ by multiplying by $1/255$.
   * Split dataset into **80% Training** and **20% Testing/Validation** sets using `ImageDataGenerator`.
3. **Model Development:** Constructed a 3-layer sequential CNN architecture with binary cross-entropy loss and Adam optimizer.
4. **Model Evaluation:** Evaluated performance over 10 training epochs using test accuracy, precision, recall, F1-score, loss/accuracy plots, and confusion matrix.

---

## CNN Architecture
* **Input Layer:** $128 \times 128 \times 3$ image input
* **Convolutional Block 1:** `Conv2D` (32 filters, $3\times3$, ReLU) $\rightarrow$ `MaxPooling2D` ($2\times2$)
* **Convolutional Block 2:** `Conv2D` (64 filters, $3\times3$, ReLU) $\rightarrow$ `MaxPooling2D` ($2\times2$)
* **Convolutional Block 3:** `Conv2D` (128 filters, $3\times3$, ReLU) $\rightarrow$ `MaxPooling2D` ($2\times2$)
* **Flattening:** `Flatten` layer converting 3D feature maps into a 1D feature vector
* **Fully Connected Layer:** `Dense` (128 neurons, ReLU activation)
* **Output Layer:** `Dense` (1 neuron, Sigmoid activation)

---

## Observations
1. **Convergence and Learning:** The training loss decreased steadily over 10 epochs while training accuracy increased, indicating proper model learning.
2. **Validation Consistency:** Validation loss and validation accuracy remained closely aligned with training metrics, showing minimal overfitting.
3. **Class Separation:** The final classification metrics demonstrate that the model successfully distinguishes distinct visual traits between cats and dogs.
4. **Feature Extraction:** Spatial features (edges, textures, shapes) extracted in early conv layers allowed the simple architecture to achieve stable validation performance.

---

## Results
* **Test Accuracy:** Evaluated after 10 epochs
* **Precision:** High proportion of accurate positive predictions
* **Recall:** Strong sensitivity in identifying true target instances
* **F1-Score:** Balanced performance across both cat and dog classes

*(Visualizations include Accuracy vs Epoch graph, Loss vs Epoch graph, and Confusion Matrix).*

---

## Conclusion
This project successfully developed a CNN model for automated cat and dog image classification. The architecture achieved robust performance by leveraging convolutional layers to extract spatial features and pooling layers to reduce dimensionality while preserving dominant traits. A major advantage of CNNs over Traditional Artificial Neural Networks (ANNs) for image classification is parameter sharing and spatial invariance, which enables the model to learn localized visual patterns regardless of position without requiring millions of unrolled pixel connections. However, a key limitation of CNNs is their heavy reliance on large amounts of labeled data and substantial compute power, making them susceptible to poor generalization when exposed to unseen background conditions or severe image distortions.
