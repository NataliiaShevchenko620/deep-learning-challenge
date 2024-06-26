# AlphabetSoupCharity Deep Learning Challenge

## Overview of the Analysis

### Purpose of the Analysis

The purpose of this analysis is to develop a machine learning model capable of predicting the success of funding applicants for Alphabet Soup, a non-profit foundation. By leveraging neural networks, the goal is to create a binary classifier that can determine whether applicants will be successful if funded. This tool aims to help Alphabet Soup make data-driven decisions in selecting applicants, thereby maximizing the impact of their funding efforts.

## Data

The dataset includes over 34,000 organizations that have received funding from Alphabet Soup. 

## Data Preprocessing

Target Variable: The target variable for the model is IS_SUCCESSFUL, which indicates whether the money was used effectively.

Feature Variables: The feature variables include APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

Removed Variables: The EIN and NAME columns were removed as they are identification columns and do not contribute to the predictive power of the model.

## Installation

To run these notebooks, you will need the following dependencies:

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- TensorFlow (or Keras)
- Scikit-learn

## Usage
1. AlphabetSoupCharity.ipynb
This notebook includes the steps for data preprocessing and initial model training:

- Data Preprocessing
- Read the dataset
- Identify target and feature variables
- Drop unnecessary columns
- Encode categorical variables
- Split data into training and testing sets
- Scale the data
- Model Training
- Create a neural network model
- Compile, train, and evaluate the model

To run this notebook, open it in Jupyter Notebook and execute all the cells.

2. AlphabetSoupCharity_Optimization.ipynb
This notebook focuses on model optimization and performance analysis:

- Data Preprocessing (repeated from the first notebook)
- Model Optimization
- Adjusting the number of neurons and layers
- Trying different activation functions
- Tuning the training process with callbacks and epochs

To run this notebook, open it in Jupyter Notebook and execute all the cells.

## Report

The analysis report includes:

Overview of the Analysis
Purpose of the Analysis
Data Preprocessing Results
Model Training and Evaluation Results
Optimization Attempts and Their Impacts


## Summary and Recommendations for Future Work

Submission

Compiling, Training, and Evaluating the Model
Neurons, Layers, and Activation Functions:

The neural network model was designed with multiple layers. The initial model included:
An input layer with nodes equal to the number of features.
Two hidden layers with ReLU activation functions.
An output layer with a sigmoid activation function to output a binary result.
Various configurations were tested, including different numbers of neurons in hidden layers and additional layers to improve model performance.
Model Performance:

The initial model achieved an accuracy of approximately 72%.
After several optimization attempts, including adjusting the number of neurons and layers, and trying different activation functions, the optimized model achieved a performance slightly higher than 75%.


![The optimized model](Image/Model.png)

Contrary to classical expectations, `adam` performs worse than `sgd`, with an `Accuracy` of 0.7521865963935852 compared to `sgd`'s `Accuracy` of 0.7526530623435974, and a `Loss` of 0.49811771512031555 compared to `sgd`'s `Loss` of 0.4973818361759186. This is a reason to study the data further.

Optimization Attempts:

- Working with data 
- Increased the number of neurons in hidden layers.
- Added more hidden layers.
- Used different activation functions such as tanh.
- Modified the number of epochs to improve training.

## Summary

The deep learning model developed for Alphabet Soup aims to predict the success of funding applicants. Although the target accuracy of 75% was achieved, the model provides a significant starting point for further enhancements. The following recommendations could help improve the model's performance:

Data Enrichment: Incorporate additional data features that could provide more insights into the success factors.
Advanced Algorithms: Experiment with more advanced machine learning algorithms such as Random Forest or Gradient Boosting.
Hyperparameter Tuning: Implement a systematic hyperparameter tuning approach using tools like GridSearchCV or Bayesian Optimization.
These steps could potentially increase the predictive accuracy and make the model a valuable tool for Alphabet Soup's funding decisions.

## Installation

1. Clone the repository to your local machine:

   ```
   [git clone https://github.com/NataliiaShevchenko620/deep-learning-challenge.git](https://github.com/NataliiaShevchenko620/deep-learning-challenge.git)
   ```

2. Install the required Python libraries
3. Run the notebook or open notebook in the Google collab and run it

## License

This project is licensed under the MIT License.