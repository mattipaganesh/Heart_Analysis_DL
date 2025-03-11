import seaborn as sns
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Keras: {}'.format(keras.__version__))

# read the csv file
data = pd.read_csv('ECG-Dataset.csv')


data.columns = ['age','sex','smoker','years_of_smoking','LDL_cholesterol','chest_pain_type','height','weight', 'familyhist',
                'activity', 'lifestyle', 'cardiac intervention', 'heart_rate', 'diabets', 'blood_pressure_sys', 'blood_pressure_dias', 
                 'hypertention', 'Interventricular_septal_end_diastole', 'ecg_pattern', 'Q_wave', 'target']


data.shape

data.head()

data.describe()

data.tail()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# Load dataset
data = pd.read_csv('ECG-Dataset.csv')

# Rename columns for easier reference
data.columns = ['age','sex','smoker','years_of_smoking','LDL_cholesterol','chest_pain_type','height','weight', 'familyhist',
                'activity', 'lifestyle', 'cardiac intervention', 'heart_rate', 'diabetes', 'blood_pressure_sys', 'blood_pressure_dias', 
                 'hypertension', 'interventricular_septal_end_diastole', 'ecg_pattern', 'Q_wave', 'target']

# Preprocessing
X = data.drop('target', axis=1)
y = data['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred).flatten()
print('Test Accuracy:', accuracy_score(y_test, y_pred_binary))
print('Classification Report:')
print(classification_report(y_test, y_pred_binary))

# Function to get user input and predict
def predict_user_input():
    user_input = []
    for i in range(X_train.shape[1]):
        user_input.append(float(input(f"Enter value for feature '{X.columns[i]}': ")))
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)
    if prediction >= 0.5:
        print("Predicted class: Heart disease present")
    else:
        print("Predicted class: No heart disease")

# Predict user input
print("\nPredicting user input:")
predict_user_input()



# Total Percentage of Missing Data
missing_data = data.isnull().sum()
total_percentage = (missing_data.sum()/data.shape[0]) * 100
print(f'The total percentage of missing data is {round(total_percentage,2)}%')
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data.hist(ax = ax)
plt.show()
# Plot Histogram to See the Distribution of the Data for Heart Disease Cases
dataset_copy=data[data['target']==1]
columns=data.columns[:21]
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
dataset_copy.hist(ax = ax)
plt.show()
sns.countplot(x='target',data=data)
plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()
data.shape
X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])
X[0]
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std
X.shape
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size = 0.2)
# convert the data to categorical labels

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=15)

model = Sequential()
model.add(Dense(64, input_dim=20, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(32, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(8, activation='softmax'))
model.add(Dropout(0.20))         
model.add(Dense(2, activation='sigmoid'))
    
# compile model
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# fit the model to the test data
history=model.fit(X_test, Y_test, validation_data=(X_test, Y_test),epochs=50, batch_size=10, callbacks=[es])

acc = model.evaluate(X_test, Y_test)
print(f"Loss:      {round(acc[0]*100,2)}%\n")
print(f"Accuracy:  {round(acc[1]*100,2)}%\n")

import matplotlib.pyplot as plt
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

# generate classification report using predictions for categorical model

categorical_pred = np.argmax(model.predict(X_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

model.save("F:\main project\model.h5")