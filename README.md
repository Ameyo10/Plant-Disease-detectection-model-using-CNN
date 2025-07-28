# Plant Disease Classification with Custom CNN

This project demonstrates how to build, train, evaluate, and deploy a Convolutional Neural Network (CNN) for plant disease classification using PyTorch and torchvision. The notebook walks through the entire deep learning workflow, from data preparation to model evaluation and prediction.

---

## 📚 What I Learned

- **PyTorch & Torchvision Libraries:**  
  Learned about essential libraries for deep learning, including how to import and use them for model building, data loading, and transformations.

- **Custom Dataset Import:**  
  Explored how to import custom datasets from the internet, organize them, and use `ImageFolder` and `DataLoader` to efficiently load and batch image data for training and validation.

- **Data Transformations:**  
  Applied various image transformations (resizing, augmentation, normalization) to improve model robustness and performance. Understood the importance of different transforms for training and validation/test sets.

- **CNN Architecture Design:**  
  Discovered how increasing the number of convolutional blocks and stacking layers in the sequence `Conv2d → BatchNorm2d → ReLU → MaxPool2d` can significantly improve model performance.

- **Linear Layer Sizing:**  
  Learned to determine the correct input size for the final linear layer in the classifier by passing a random image through the convolutional blocks and checking the output shape.

- **Training & Validation Loops:**  
  Built modular `train_step()` and `valid_step()` functions, and combined them into a unified `train()` function for streamlined training and evaluation.

- **Model Checkpointing:**  
  Implemented logic to check if a trained model already exists. If so, the model is loaded and evaluated; if not, it is trained, evaluated, and then saved for future use.

- **Performance Visualization:**  
  Plotted loss and accuracy curves to monitor training progress, and generated a confusion matrix to analyze model predictions and class-wise performance.

- **Test Set Prediction:**  
  Used the trained model to predict on unseen test images, achieving significant success and demonstrating the model's practical utility.

---

## 🏆 Results & Outputs

- **High Validation Accuracy:**  
  The final CNN model achieved strong validation accuracy, indicating effective learning and generalization.

- **Loss & Accuracy Curves:**  
  Plotted curves showed steady improvement and convergence during training.

- **Confusion Matrix:**  
  The confusion matrix revealed that the model predicts most classes accurately, with minimal confusion between similar diseases.

- **Test Predictions:**  
  The model performed well on the test set, correctly identifying plant diseases in most cases.

---

## 🚀 How to Use

1. **Clone the Repository**
2. **Install Requirements:**  
   ```bash
   pip install torch torchvision torchmetrics mlxtend matplotlib pandas
   ```
3. **Prepare Dataset:**  
   Download and organize your dataset as shown in the notebook.
4. **Run the Notebook:**  
   Open the notebook in VS Code or Jupyter and run all cells.

---

## 📈 Example Outputs

- **Loss & Accuracy Curves:**  
  ![Loss and Accuracy Curves](https://github.com/Ameyo10/Plant-Disease-detectection-model-using-CNN/blob/main/Loss%20and%20Accuracy%20curve%20of%20my%20model_training.png)

- **Confusion Matrix:**  
  ![Confusion Matrix](https://github.com/Ameyo10/Plant-Disease-detectection-model-using-CNN/blob/main/Confusion%20Matrix%20of%20my%20model.png)

---

## 📝 Conclusion

This project demonstrates the full workflow of deep learning for image classification, from data loading and augmentation to model design, training, evaluation, and deployment. The experience gained here can be applied to a wide range of computer vision tasks.

---
```# Plant Disease Classification with Custom CNN

This project demonstrates how to build, train, evaluate, and deploy a Convolutional Neural Network (CNN) for plant disease classification using PyTorch and torchvision. The notebook walks through the entire deep learning workflow, from data preparation to model evaluation and prediction.

---

## 📚 What I Learned

- **PyTorch & Torchvision Libraries:**  
  Learned about essential libraries for deep learning, including how to import and use them for model building, data loading, and transformations.

- **Custom Dataset Import:**  
  Explored how to import custom datasets from the internet, organize them, and use `ImageFolder` and `DataLoader` to efficiently load and batch image data for training and validation.

- **Data Transformations:**  
  Applied various image transformations (resizing, augmentation, normalization) to improve model robustness and performance. Understood the importance of different transforms for training and validation/test sets.

- **CNN Architecture Design:**  
  Discovered how increasing the number of convolutional blocks and stacking layers in the sequence `Conv2d → BatchNorm2d → ReLU → MaxPool2d` can significantly improve model performance.

- **Linear Layer Sizing:**  
  Learned to determine the correct input size for the final linear layer in the classifier by passing a random image through the convolutional blocks and checking the output shape.

- **Training & Validation Loops:**  
  Built modular `train_step()` and `valid_step()` functions, and combined them into a unified `train()` function for streamlined training and evaluation.

- **Model Checkpointing:**  
  Implemented logic to check if a trained model already exists. If so, the model is loaded and evaluated; if not, it is trained, evaluated, and then saved for future use.

- **Performance Visualization:**  
  Plotted loss and accuracy curves to monitor training progress, and generated a confusion matrix to analyze model predictions and class-wise performance.

- **Test Set Prediction:**  
  Used the trained model to predict on unseen test images, achieving significant success and demonstrating the model's practical utility.

---

## 🏆 Results & Outputs

- **High Validation Accuracy:**  
  The final CNN model achieved strong validation accuracy, indicating effective learning and generalization.

- **Loss & Accuracy Curves:**  
  Plotted curves showed steady improvement and convergence during training.

- **Confusion Matrix:**  
  The confusion matrix revealed that the model predicts most classes accurately, with minimal confusion between similar diseases.

- **Test Predictions:**  
  The model performed well on the test set, correctly identifying plant diseases in most cases.

---

## 🚀 How to Use

1. **Clone the Repository**
2. **Install Requirements:**  
   ```bash
   pip install torch torchvision torchmetrics mlxtend matplotlib pandas
   ```
3. **Prepare Dataset:**  
   Download and organize your dataset as shown in the notebook.
4. **Run the Notebook:**  
   Open the notebook in VS Code or Jupyter and run all cells.

---

## 📈 Example Outputs

- **Loss & Accuracy Curves:**  
  ![Loss and Accuracy Curves](https://github.com/Ameyo10/Plant-Disease-detectection-model-using-CNN/blob/main/Loss%20and%20Accuracy%20curve%20of%20my%20model_training.png)

- **Confusion Matrix:**  
  ![Confusion Matrix]()

---

## 📝 Conclusion

This project demonstrates the full workflow of deep learning for image classification, from data loading and augmentation to model design, training, evaluation, and deployment. The experience gained here can be applied to a wide range of computer vision
