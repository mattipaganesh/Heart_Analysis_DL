from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model
#from keras.utils.np_utils import to_categorical

#from tensorflow.contrib.keras.python.keras.utils    import np_utils

from tensorflow.python.keras import utils

app = Flask(__name__)


# Load the model
model = load_model('model.h5')


# Load dataset
data = pd.read_csv('ECG-Dataset.csv')
data.columns = ['age','sex','smoker','years_of_smoking','LDL_cholesterol','chest_pain_type','height','weight', 'familyhist',
                'activity', 'lifestyle', 'cardiac intervention', 'heart_rate', 'diabetes', 'blood_pressure_sys', 'blood_pressure_dias', 
                 'hypertension', 'interventricular_septal_end_diastole', 'ecg_pattern', 'Q_wave', 'target']

# Preprocessing
X = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = [float(request.form[f'input_{i+1}']) for i in range(X_scaled.shape[1])]
        user_input_scaled = scaler.transform([user_input])

        # Make prediction
        prediction = model.predict(user_input_scaled)
        if prediction >= 0.5:
            prediction_text = "Heart disease present"
        else:
            prediction_text = "No heart disease"

        # Create plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        sns.histplot(data['age'], ax=axes[0, 0])
        sns.countplot(x='sex', data=data, ax=axes[0, 1])
        sns.heatmap(data.corr(), annot=True, fmt='.1f', ax=axes[1, 0])
        pd.crosstab(data['age'], data['target']).plot(kind="bar", figsize=(15, 5), ax=axes[1, 1])
        axes[1, 1].set_title('Heart Disease Frequency for Ages')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend(['No Disease', 'Disease'])
        axes[2, 0].text(0.5, 0.5, prediction_text, horizontalalignment='center', verticalalignment='center', fontsize=20)
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        
        # Save plots to a temporary file
        temp_file = 'static/plots.png'
        plt.savefig(temp_file)
        plt.close()

        return render_template('result.html', prediction=prediction_text, plot=temp_file)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    
    # Standardize the input features
    final_features = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(final_features)
    prediction = (prediction > 0.5).astype(int)[0][0]
    
    # Generate prediction text
    if prediction == 1:
        prediction_text = "The model predicts that the patient has heart disease."
    else:
        prediction_text = "The model predicts that the patient does not have heart disease."
   

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()
    # Save the heatmap as an image
    heatmap_path = 'static/heatmap.png'
    plt.savefig(heatmap_path)
    #plt.close()
    # Generate the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['age'], data['hr'], color='blue', alpha=0.5)
    plt.title('Scatter Plot of Age vs. Heart Rate')
    plt.xlabel('Age')
    plt.ylabel('Heart Rate')
    plt.tight_layout()

    # Save the scatter plot as an image
    scatter_path = 'static/scatter_plot.png'
    plt.savefig(scatter_path)
    plt.close()
    # Generate the bar chart
    plt.figure(figsize=(10, 6))
    data['sex'].value_counts().plot(kind='bar', color='green')
    plt.title('Bar Chart of Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    # Save the bar chart as an image
    bar_chart_path = 'static/bar_chart.png'
    plt.savefig(bar_chart_path)
    plt.close()
            
    # Generate the pie chart
    plt.figure(figsize=(8, 8))
    data['smoke'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange'])
    plt.title('Pie Chart of Smoker Distribution')
    plt.ylabel('')
    plt.tight_layout()

    # Save the pie chart as an image
    pie_chart_path = 'static/pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()
    # Generate the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['age'], data['weight'], marker='o', color='red', linestyle='-')
    plt.title('Line Plot of Age vs. Weight')
    plt.xlabel('Age')
    plt.ylabel('Weight')
    plt.tight_layout()

    # Save the line plot as an image
    line_plot_path = 'static/line_plot.png'
    plt.savefig(line_plot_path)
    plt.close()
    # Generate the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data['height'], bins=20, color='purple', alpha=0.7)
    plt.title('Histogram of Height')
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the histogram as an image
    histogram_path = 'static/histogram.png'
    plt.savefig(histogram_path)
    plt.close()

    # Save individual plots
    plot_files = []
    for i, feature in enumerate(features):
        plt.figure()
        plt.bar([0], [feature])
        plt.xlabel(f'Feature {i+1}')
        plt.ylabel('Value')
        plt.title(f'Feature {i+1}')
        plot_path = f'static/plot_{i+1}.png'
        plt.savefig(plot_path)
        plt.close()
        plot_files.append(plot_path)
    
    return render_template('result.html', prediction_text=prediction_text, plots=plot_files)


if __name__ == '__main__':
    app.run(debug=True)


