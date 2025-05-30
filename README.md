# Bank Churn Prediction Analysis with Machine Learning

## Project Overview

This project aims to develop a robust machine learning model to predict customer churn in a banking context, specifically for loan customers. By identifying customers who are likely to churn, banks can proactively implement retention strategies, thereby reducing customer attrition and improving overall customer lifetime value.

## Features

* **Data Preprocessing:** Handling of categorical features, scaling numerical data, and preparing the dataset for model training.

* **Exploratory Data Analysis (EDA):** Visualizations and statistical summaries to understand customer demographics, account behavior, and their relationship with churn.

* **Multiple Machine Learning Models:** Experimentation and comparison of various classification algorithms to find the best-performing model for churn prediction.

* **Model Evaluation:** Comprehensive evaluation using metrics such as classification reports, AUC scores, and various plots to assess model performance.

* **Model Persistence:** Saving the trained optimal model for future use and deployment.

* **Web Application (Flask):** A simple web interface to demonstrate the churn prediction model.

## Dataset

The dataset used for this project is sourced from Kaggle: [Bank Customer Churn Prediction](https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction/input).

It typically contains features related to customer demographics (e.g., `CreditScore`, `Geography`, `Gender`, `Age`), account details (e.g., `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`), and the target variable indicating churn (`Exited`).

## Technologies Used

This project leverages the following technologies and Python libraries:

* **Python:** Programming language

* **Jupyter Notebook / Google Colab:** For interactive development, experimentation and analysis.

* **Pandas:** For data manipulation and analysis.

* **NumPy:** For numerical operations.

* **Matplotlib:** For static data visualization.

* **Seaborn:** For enhanced statistical data visualization.

* **Scikit-learn:** For machine learning model building, evaluation, and preprocessing utilities.

* **XGBoost:** For implementing the Extreme Gradient Boosting algorithm.

* **Joblib:** For saving and loading Python objects, specifically the trained machine learning model.

* **Flask:** For building the lightweight web application to serve predictions.

## Project Structure

The project is organized into the following directories and files:

```
.
├── .idea/                      # IDE-specific configuration (e.g., PyCharm)
├── .venv/                      # Python virtual environment
├── templates/                  # HTML templates for the Flask application
│   ├── index1.html             # First main page for the web application
│   ├── index2.html             # Second main page for the web application
│   └── index3.html             # Third main page for the web application
├── app.py                      # Flask application to serve predictions
├── bankchurn.csv               # The raw dataset used for the project
├── bank_customer_churn_prediction_FINAL.ipynb # Jupyter Notebook containing EDA, model training, and evaluation
├── best_xgboost_model.pkl      # The best-performing XGBoost model saved using joblib
├── requirements.txt            # List of Python dependencies
└── test.py                     # (Optional) Script for testing specific functionalities
```

## Machine Learning Models Explored

The `bank_customer_churn_prediction_FINAL.ipynb` notebook includes experimentation and evaluation of the following classification models:

* **Logistic Regression (Primal):** A baseline linear model.

* **Logistic Regression with Degree 2 Polynomial Kernel:** Exploring non-linear relationships.

* **Support Vector Machine (SVM) with RBF Kernel:** A powerful non-linear classifier.

* **Support Vector Machine (SVM) with Polynomial Kernel:** Another non-linear SVM variant.

* **Random Forest Classifier:** An ensemble tree-based method.

* **Extreme Gradient Boosting (XGBoost) Classifier:** A highly efficient and effective gradient boosting framework, which was identified as the best-fit model for this problem.

For each model, the notebook includes:

* Model fitting and training.

* Hyperparameter tuning (if applicable).

* Generation of `classification_report`.

* Plotting of relevant graphs (e.g., ROC curves, confusion matrices).

* Calculation of AUC (Area Under the Curve) scores.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/VishaL6i9/bank-churn-prediction-analysis-ml.git
    cd bank-churn-prediction-analysis-ml
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    * **On Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    * **On macOS/Linux:**

        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required dependencies:**

    The `requirements.txt` file specifies the exact versions of the libraries used:

    ```
    numpy~=2.2.4
    pandas~=2.2.3
    scikit-learn~=1.6.1
    matplotlib~=3.10.1
    seaborn~=0.13.2
    joblib~=1.4.2
    Flask~=3.1.0
    xgboost==2.1.4
    ```

    Install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Jupyter Notebook

To explore the data analysis, model training, and evaluation process, open the Jupyter Notebook:

```bash
jupyter notebook bank_customer_churn_prediction_FINAL.ipynb
```

### Running the Flask Web Application

To run the Flask application and make predictions via a web interface:

1.  Ensure your virtual environment is activated.

2.  Navigate to the project's root directory.

3.  Run the `app.py` script:

    ```bash
    python app.py
    ```

4.  Open your web browser and go to `http://127.0.0.1:5000/` (or the address shown in your terminal) to access the application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or suggestions, feel free to reach out:

* **Vishal Kandakatla —** [vishalkandakatla@gmail.com]
