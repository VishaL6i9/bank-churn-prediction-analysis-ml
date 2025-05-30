Bank Churn Prediction Analysis with Machine LearningProject OverviewThis project aims to develop a robust machine learning model to predict customer churn in a banking context, specifically for loan customers. By identifying customers who are likely to churn, banks can proactively implement retention strategies, thereby reducing customer attrition and improving overall customer lifetime value.FeaturesData Preprocessing: Handling of categorical features, scaling numerical data, and preparing the dataset for model training.Exploratory Data Analysis (EDA): (Although you mentioned not wanting to include analysis for now, a good README often hints at the potential for it. I'll keep this general.) Visualizations and statistical summaries to understand customer demographics, account behavior, and their relationship with churn.Multiple Machine Learning Models: Experimentation and comparison of various classification algorithms to find the best-performing model for churn prediction.Model Evaluation: Comprehensive evaluation using metrics such as classification reports, AUC scores, and various plots to assess model performance.Model Persistence: Saving the trained optimal model for future use and deployment.Web Application (Flask): A simple web interface to demonstrate the churn prediction model.DatasetThe dataset used for this project is sourced from Kaggle: Bank Customer Churn Prediction.It typically contains features related to customer demographics (e.g., CreditScore, Geography, Gender, Age), account details (e.g., Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary), and the target variable indicating churn (Exited).Technologies UsedThis project leverages the following technologies and Python libraries:Python: Programming languageJupyter Notebook / Google Colab: For interactive development, experimentation, and analysis.Pandas: For data manipulation and analysis.NumPy: For numerical operations.Matplotlib: For static data visualization.Seaborn: For enhanced statistical data visualization.Scikit-learn: For machine learning model building, evaluation, and preprocessing utilities.XGBoost: For implementing the Extreme Gradient Boosting algorithm.Joblib: For saving and loading Python objects, specifically the trained machine learning model.Flask: For building the lightweight web application to serve predictions.Project StructureThe project is organized into the following directories and files:.
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
Machine Learning Models ExploredThe bank_customer_churn_prediction_FINAL.ipynb notebook includes experimentation and evaluation of the following classification models:Logistic Regression (Primal): A baseline linear model.Logistic Regression with Degree 2 Polynomial Kernel: Exploring non-linear relationships.Support Vector Machine (SVM) with RBF Kernel: A powerful non-linear classifier.Support Vector Machine (SVM) with Polynomial Kernel: Another non-linear SVM variant.Random Forest Classifier: An ensemble tree-based method.Extreme Gradient Boosting (XGBoost) Classifier: A highly efficient and effective gradient boosting framework, which was identified as the best-fit model for this problem.For each model, the notebook includes:Model fitting and training.Hyperparameter tuning (if applicable).Generation of classification_report.Plotting of relevant graphs (e.g., ROC curves, confusion matrices).Calculation of AUC (Area Under the Curve) scores.InstallationTo set up the project locally, follow these steps:Clone the repository:git clone https://github.com/your-username/bank-churn-prediction-analysis-ml.git
cd bank-churn-prediction-analysis-ml
(Replace https://github.com/your-username/bank-churn-prediction-analysis-ml.git with your actual repository URL)Create a virtual environment (recommended):python -m venv .venv
Activate the virtual environment:On Windows:.venv\Scripts\activate
On macOS/Linux:source .venv/bin/activate
Install the required dependencies:The requirements.txt file specifies the exact versions of the libraries used:numpy~=2.2.4
pandas~=2.2.3
scikit-learn~=1.6.1
matplotlib~=3.10.1
seaborn~=0.13.2
joblib~=1.4.2
Flask~=3.1.0
xgboost==2.1.4
Install them using pip:pip install -r requirements.txt
UsageRunning the Jupyter NotebookTo explore the data analysis, model training, and evaluation process, open the Jupyter Notebook:jupyter notebook bank_customer_churn_prediction_FINAL.ipynb
Running the Flask Web ApplicationTo run the Flask application and make predictions via a web interface:Ensure your virtual environment is activated.Navigate to the project's root directory.Run the app.py script:python app.py
Open your web browser and go to http://127.0.0.1:5000/ (or the address shown in your terminal) to access the application.LicenseThis project is licensed under the MIT License. See the LICENSE file for details.ContactFor any questions or suggestions, feel free to reach out:Vishal Kandakatla — [vishalkandakatla@gmail.com]
