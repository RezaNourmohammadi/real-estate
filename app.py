import os
import pickle
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import json
from flask import json




app = Flask(__name__)

# Define the folder paths
DATA_FOLDER = os.path.join(app.root_path, 'data')
SAVED_MODELS_FOLDER = os.path.join(app.root_path, 'saved_models')

# Load the dataset
data_path = os.path.join(DATA_FOLDER, 'Data.csv')
data = pd.read_csv(data_path)


# Assuming df is your DataFrame and 'area' is the column you want to convert
data = data[data['Area'] != 'q']
data['Area'] = data['Area'].astype(str).astype(int)

data['Document'] = data['Document'].fillna(-1)
data['Floor'] = data['Floor'].fillna(0)
data['Reconstruction'] = data['Reconstruction'].fillna(0)
data['Warehouse'] = data['Warehouse'].fillna(0)
data['Document'] = data['Document'].fillna(-1)
data['Parking'] = data['Parking'].fillna(0)
data['Warehouse'] = data['Warehouse'].fillna(0)



@app.route('/get_models', methods=['GET'])
def get_models():
    # Get the list of model filenames in the saved_models folder
    model_files = os.listdir(SAVED_MODELS_FOLDER)
    # Extract model names from filenames
    models = [filename.split('.')[0] for filename in model_files if filename.endswith('.pkl')]
    return jsonify({'models': models})

def preprocess_data(df):
    # Split the data into features and target
    label_encoder = LabelEncoder()
    encoding_schema = {}
    unique_values = {}
    
    

    # Encode categorical features
    for column in df.columns:
        if df[column].dtype == "object" and column != "TotalPrice":
            # encoded_column = label_encoder.fit_transform(df[column])
            # df[column] = encoded_column
            # encoding_schema[column] = {str(key): str(value) for key, value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            unique_values[column] = df[column].unique().tolist()

    # Save encoding schema for numerical features (if needed)
    # for column in df.columns:
    #     if df[column].dtype != "object" and column != "total_value" and column not in encoding_schema:
    #         encoding_schema[column] = None  # Placeholder for numerical features

    for column in df.columns:
        if column != "TotalPrice" and column not in unique_values:  # Skip already processed columns
            unique_values[column] = df[column].unique().tolist()

    x = df.drop("TotalPrice", axis=1)
    y = df['TotalPrice']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test, encoding_schema, unique_values

# Train the regression models
def train_model(model_name, X_train, y_train):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Lasso Regression':
        model = Lasso()
    elif model_name == 'SVR':
        model = SVR()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor()
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor()
    elif model_name=='DecisionTree':
        model=DecisionTreeRegressor(random_state = 42)
    elif model_name=='rbf':
        model = SVR(kernel='rbf')
    else:
        return 'Invalid model selected'

    # Train the model on the training set
    model.fit(X_train, y_train)

    return model

@app.route('/get_features', methods=['POST'])
def get_features():
    model_name = request.json['model_name']
    features_file = os.path.join(SAVED_MODELS_FOLDER, f"{model_name}_unique_values.json")

    if not os.path.exists(features_file):
        return jsonify({'message': f'Features file for model "{model_name}" not found'}), 404

    with open(features_file, 'r') as f:
        features_data = json.load(f)
        print(jsonify({'features': features_data}))
    return json.dumps({'features': features_data},sort_keys=False)


@app.route('/test_model', methods=['POST'])
def test_model():
    selected_model = request.json['selected_model']
    model_filename = selected_model + '.pkl'
    model_path = os.path.join(SAVED_MODELS_FOLDER, model_filename)

    if not os.path.exists(model_path):
        return jsonify({'message': f'Model "{selected_model}" not found'}), 404

    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load test data
    test_filename = selected_model + '_test.csv'
    test_path = os.path.join(SAVED_MODELS_FOLDER, test_filename)
    test_data = pd.read_csv(test_path)

    # Prepare test data
    X_test = test_data.drop(columns=['TotalPrice'])
    y_test = test_data['TotalPrice']

    y_pred = model.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    return jsonify({'message': f'R2 score: {r2:.4f}, MSE: {mse:.4f}'})

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model and user input from the request
    selected_model = request.json['selected_model']
    user_input = request.json['user_input']

    # Load the selected model
    model_file = os.path.join(SAVED_MODELS_FOLDER, selected_model + '.pkl')
    if not os.path.exists(model_file):
        return jsonify({'message': f'Model "{selected_model}" not found'}), 404

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Load the encoding schema
    # encoding_schema_file = os.path.join(SAVED_MODELS_FOLDER, selected_model + '_encoding_schema.json')
    # if not os.path.exists(encoding_schema_file):
    #     return jsonify({'message': f'Encoding schema file for model "{selected_model}" not found'}), 404

    # with open(encoding_schema_file, 'r') as f:
    #     encoding_schema = json.load(f)

    # Prepare input array for prediction
    input_array = []
    # for feature, value in user_input.items():
    #     if feature in encoding_schema and encoding_schema[feature] is not None:
    #         encoded_value = encoding_schema[feature].get(str(value), None)
    #         if encoded_value is not None:
    #             input_array.append(encoded_value)
    #         else:
    #             return jsonify({'message': f'Invalid value "{value}" for feature "{feature}"'}), 400
    #     else:
    #         # If encoding schema is None or feature is not in encoding schema, use the original value
    #         input_array.append(value)
    
    for feature, value in user_input.items():
         input_array.append(int(value))

    input_array = np.array([input_array])
    
    print(input_array)

    # Perform prediction
    prediction = model.predict(input_array)

    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})



# Save the trained model
def save_model(model, model_name):
    model_filename = model_name + '.pkl'
    model_path = os.path.join(SAVED_MODELS_FOLDER, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# Save the train and test data
def save_data(X_train, X_test, y_train, y_test, model_name):
    train_filename = model_name + '_train.csv'
    test_filename = model_name + '_test.csv'
    train_path = os.path.join(SAVED_MODELS_FOLDER, train_filename)
    test_path = os.path.join(SAVED_MODELS_FOLDER, test_filename)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


@app.route('/train', methods=['POST'])
def train():
    # Get the selected model and model name from the request
    selected_model = request.json['selected_model']
    model_name = request.json['model_name']

    # Preprocess the data
    X_train, X_test, y_train, y_test, encoding_schema, unique_values = preprocess_data(data)

    # Train the model
    trained_model = train_model(selected_model, X_train, y_train)

    if trained_model == 'Invalid model selected':
        return jsonify({'message': 'Invalid model selected'})

    # Save the trained model
    save_model(trained_model, model_name)
    save_data(X_train, X_test, y_train, y_test, model_name)
    # save_encoding_schema(encoding_schema, model_name)
    save_unique_values(unique_values, model_name)

    # Return a message indicating that training is finished
    return jsonify({'message': 'Training finished'})

def save_encoding_schema(encoding_schema, model_name):
    encoding_schema_file = os.path.join(SAVED_MODELS_FOLDER, model_name + '_encoding_schema.json')
    with open(encoding_schema_file, 'w') as f:
        json.dump(encoding_schema, f)
        
def save_unique_values(unique_values, model_name):
    unique_values_file = os.path.join(SAVED_MODELS_FOLDER, model_name + '_unique_values.json')
    with open(unique_values_file, 'w') as f:
        json.dump(unique_values, f)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
