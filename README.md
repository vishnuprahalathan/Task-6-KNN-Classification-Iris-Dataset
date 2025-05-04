This project demonstrates a complete machine learning pipeline using the K-Nearest Neighbors (KNN) algorithm on the classic Iris dataset. It includes data preprocessing, model training with cross-validation, evaluation, and 2D decision boundary visualization.

---

Dataset

The Iris dataset contains 150 samples of iris flowers, classified into three species:
- Setosa
- Versicolor
- Virginica

Each sample has 4 features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
 [Download Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

---

 Libraries Used

- `pandas` for data loading and analysis  
- `numpy` for numerical operations  
- `matplotlib` & `seaborn` for data visualization  
- `scikit-learn` for preprocessing, modeling, and evaluation  

---

Ensure you have all dependencies installed:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn
Place Iris.csv in the same directory.

Features
Loads and analyzes the Iris dataset
Encodes target labels
Splits data into training and testing sets
Applies standardization
Performs KNN classification with K from 1 to 20
Selects the best K using 5-fold cross-validation
Evaluates the final model using accuracy, confusion matrix, and classification report
Visualizes decision boundaries in 2D feature space


Results
Best K is chosen based on highest CV accuracy

Test Accuracy is printed for evaluation

Classification Report includes precision, recall, and F1-score

Decision Boundary helps understand how KNN separates classes

