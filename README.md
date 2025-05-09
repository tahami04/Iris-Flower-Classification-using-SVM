# Iris-Flower-Classification-using-SVM
This project demonstrates iris flower classification using a Support Vector Machine (SVM) in Python. It includes data visualization, model training, hyperparameter tuning, evaluation with cross-validation, and prediction on sample inputs using scikit-learn and pandas.

**Project Overview**
This project uses the famous Iris dataset to demonstrate supervised classification techniques. The steps include loading the dataset into a Pandas DataFrame, visualizing feature relationships (e.g. scatter plots of sepal measurements), and training an SVM classifier on the data. The notebook shows how to split the data into training and testing sets, fit the SVM model, and make predictions. It also explores the effects of changing parameters such as the regularization strength (C) and kernel type. Finally, the model is evaluated by computing its accuracy on the test set, displaying a confusion matrix, and performing k-fold cross-validation to measure average performance. This end-to-end example illustrates basic machine learning workflow components: data preparation, training, and evaluation.

**Features**
**Data Loading & Exploration:** Loads the Iris dataset (sepal and petal measurements for three iris species) using scikit-learn and creates a Pandas DataFrame for inspection.
**Data Visualization:** Plots features (e.g. sepal length vs. sepal width) for different iris species to visualize class separability.
**SVM Classification:** Trains a Support Vector Machine (SVC) classifier to predict iris species.
**Hyperparameter Tuning:** Demonstrates adjusting SVM parameters like the regularization parameter C and kernel choice, and shows their impact on accuracy.
**Model Evaluation:** Computes model accuracy on a test set, displays a confusion matrix, and uses 10-fold cross-validation to assess stability (mean accuracy and standard deviation).
**Sample Prediction:** Includes example predictions on new sample data points to show how to use the trained model for inference.

**Installation and Setup**
Clone the repository (or download the notebook file):

git clone https://github.com/<username>/iris-svm-classification.git
cd iris-svm-classification

**Install dependencies:** Ensure you have Python 3 installed. Then install the required Python libraries:

pip install pandas scikit-learn matplotlib
Run the Notebook: Launch Jupyter Notebook or Jupyter Lab, and open the ML L12.ipynb file. Run the cells sequentially to reproduce the analysis.

**Example Usage**
Below is an example of how to train an SVM classifier on the Iris dataset and evaluate its accuracy (this parallels the steps in the notebook):
python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(C=1.0, kernel='rbf')  # default C=1.0, RBF kernel
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
You can also print a confusion matrix to see class-wise performance:

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

**Dataset**
This project uses the Iris dataset, a classic dataset included with scikit-learn. It contains 150 samples of iris flowers across three species (setosa, versicolor, virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width (all in centimeters). The dataset is loaded via sklearn.datasets.load_iris, which returns both the feature matrix and target labels. In the notebook, the data is also converted to a Pandas DataFrame for easier handling and viewing of feature names. The Iris dataset originally comes from the UCI Machine Learning Repository and is commonly used for testing classification algorithms.

**Contributing**
This is a personal project demonstration. However, contributions and improvements are welcome! Feel free to fork the repository and submit pull requests if you find enhancements (such as additional visualizations, more models, or documentation). You can also open an issue to report bugs or suggest features.
