# Iris Dataset Machine Learning Model

This project demonstrates how to create a machine learning model using the iris dataset. The model is trained using a RandomForestClassifier and deployed using Streamlit.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/varun2388/mlmodelrepo.git
   cd mlmodelrepo
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Model Training Script

To train the machine learning model on the iris dataset, run the following command:
```
python model.py
```
This will load the iris dataset, split it into training and testing sets, train a RandomForestClassifier, evaluate the model, and save the trained model to a file. The script will only train the model if the `model.joblib` file does not exist.

## Deploying the Model using Streamlit

To deploy the trained model using Streamlit, run the following command:
```
streamlit run app.py
```
This will start a Streamlit app with input fields for the iris dataset features and display the predicted class for the input features using the trained model.
