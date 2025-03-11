from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = Flask(__name__)

# Load the model and the scaler
model = tf.keras.models.load_model('heart_disease_model.h5')  # Ensure the model path is correct
scaler = joblib.load('scaler.save')
data = pd.read_csv('ECG-Dataset.csv')
# Ensure the 'static' directory exists to save plots
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def home():
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

if __name__ == "__main__":
    app.run(debug=True)
