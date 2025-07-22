# Real-estate-price-predection
🏠 Real Estate Price Prediction
A machine learning model that predicts real estate prices based on property features such as area, location, number of rooms, and more. Built using Python, Scikit-learn, Pandas, and Matplotlib. Includes a visualization of area vs price and basic data preprocessing for categorical inputs.

📌 Table of Contents
Features

Tech Stack

Project Structure

Getting Started

Usage

Sample Prediction

Results & Evaluation

Future Work

✅ Features
Preprocesses real estate data (categorical + numerical).

Trains a Linear Regression model to predict house prices.

Accepts custom property specifications as input.

Displays price predictions.

Graphs Area vs Price for visual analysis.

Evaluates model performance using MAE, MSE, RMSE, and R² score.

🛠 Tech Stack
Python

Pandas

Scikit-learn

Matplotlib

NumPy

📁 Project Structure
csharp
Copy
Edit
├── Data set for this project.csv     # Dataset file
├── real_estate_predictor.py         # Main Python script
├── README.md                        # Project documentation
⚙️ Getting Started
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/real-estate-price-predictor.git
cd real-estate-price-predictor
2. Install dependencies
Make sure you have Python 3.x installed, then run:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib
3. Add dataset
Place your dataset file as Data set for this project.csv in the root directory.
It should contain columns like: area, bedrooms, bathrooms, location, parking_space, age_of_building, furnishing_status, and price.

▶️ Usage
Run the main script:

bash
Copy
Edit
python real_estate_predictor.py
This will:

Train the regression model

Predict the price for a sample property

Print model evaluation metrics

Plot a graph of Area vs Price

📊 Sample Prediction
python
Copy
Edit
input_specs = {
    "area": [2963],
    "bedrooms": [1],
    "bathrooms": [3],
    "location": ["Uptown"],
    "parking_space": [0],
    "age_of_building": [6],
    "furnishing_status": ["Furnished"]
}
Output:

yaml
Copy
Edit
Predicted Price: 7550000.0

Model Evaluation Metrics:
MAE: 238456.19
MSE: 1432459805.34
RMSE: 37834.71
R² Score: 0.9132
📈 Results & Evaluation
Model is evaluated using:

✅ Mean Absolute Error (MAE)

✅ Mean Squared Error (MSE)

✅ Root Mean Squared Error (RMSE)

✅ R² Score

Also includes a plot of actual vs predicted values using matplotlib.

🚀 Future Work
Use advanced models like Random Forest or XGBoost

Add support for more feature engineering

Build a web app with Streamlit or Flask

Improve preprocessing with Label Encoding or One-Hot Encoding

Deploy using Docker or Render

