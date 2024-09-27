# Major_Project
This is a repository which consist of my CDAC PGDBDA Final Project Title-"Human Activity Recognition"

Human Activity Recognition (HAR) is a machine learning project aimed at identifying and classifying different physical activities performed by a person based on data collected from wearable sensors like accelerometers, gyroscopes, and other devices. This technology has applications in health monitoring, fitness tracking, and smart environments.

Key Components of the Project:
Data Collection:

Data is usually collected from wearable devices like smartphones, smartwatches, or dedicated fitness trackers.
These devices capture sensor data such as accelerometer (measures acceleration), gyroscope (measures orientation and angular velocity), and sometimes additional sensors like magnetometers or GPS.
Each sensor records data in multiple dimensions (e.g., X, Y, Z axes for acceleration), and this data is collected at regular intervals.
Data Preprocessing:

Cleaning: Removing noisy or irrelevant data.
Normalization/Standardization: Ensuring that the data is in a format suitable for model training.
Segmentation: Dividing time-series data into fixed-length windows to capture movement patterns over time.
Feature extraction: Extracting meaningful features from raw sensor data, like mean, variance, and signal magnitudes, which help classify activities more accurately.
Activity Classes: The project typically focuses on recognizing activities such as:

Walking
Running
Sitting
Standing
Laying down
Climbing stairs
Cycling
Model Selection: Machine learning models are trained to classify these activities based on the processed sensor data. Popular models include:

Traditional algorithms: Decision Trees, Random Forests, Support Vector Machines (SVM), k-Nearest Neighbors (k-NN).
Deep Learning models: Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) (especially LSTMs) are often used to capture the temporal dependencies and patterns in the data.
Training and Testing:

The collected dataset is split into training and testing sets.
The model is trained on the training set to learn how to classify human activities based on sensor input.
The testing set is used to evaluate the model’s performance in recognizing unseen activities.
Evaluation Metrics: Common metrics to evaluate the model’s performance include:

Accuracy: The percentage of correctly classified activities.
Precision, Recall, F1-score: To assess how well the model performs on each activity class.
Confusion Matrix: Helps visualize the true vs predicted activities.
Applications:

Healthcare: Monitoring elderly people or patients for fall detection or irregular activity patterns.
Fitness: Activity tracking and calorie estimation for fitness enthusiasts.
Smart Homes: Automated systems that adapt to the user's behavior and preferences.
Security: Detecting suspicious or unusual activity in surveillance systems.
Project Workflow:
Collect sensor data from wearable devices or use publicly available datasets.
Preprocess the data by cleaning, normalizing, and segmenting it into windows.
Feature extraction from the raw sensor data or feed it directly into deep learning models.
Model training using machine learning algorithms to classify the activities.
Evaluate the model on a test set and tune the parameters for better accuracy.
Deploy the model in real-time or offline for human activity recognition in applications like health monitoring or fitness tracking.
