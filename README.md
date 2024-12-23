# Healthcare-Analytics-Toolset
A comprehensive machine learning project focused on breast cancer detection and healthcare cost prediction. Features multiple models including CNN image classification for FNA samples, decision tree analysis for patient data, and linear regression for insurance cost forecasting. Built with TensorFlow, scikit-learn, and pandas.

# Project1
This component implements a linear regression model to predict healthcare insurance costs based on various patient factors.
Features:

Processes demographic and health-related data
Converts categorical variables (sex, smoking status, region) to numerical values
Implements linear regression for cost prediction
Provides detailed feature importance analysis
Visualizes predictions vs actual costs
Generates coefficient analysis for factor impact

Dependencies:

pandas
numpy
scikit-learn
matplotlib

# Project2
A decision tree-based classification system for breast cancer diagnosis prediction using patient data.
Features:

Handles multiple categorical variables (Race, Marital Status, Cancer Stages)
Implements decision tree classifier with depth optimization
Provides performance metrics (accuracy, log loss)
Generates visual decision tree representation
Includes model performance visualization
Cross-validates results

Dependencies:

pandas
numpy
scikit-learn
matplotlib

# Project3
A Convolutional Neural Network (CNN) based system for analyzing Fine Needle Aspiration (FNA) images for breast cancer detection.
Features:

Processes and normalizes medical images
Implements CNN architecture for image classification
Provides real-time training monitoring
Generates performance visualizations
Includes prediction capability for new images
Saves trained models for future use

Dependencies:

TensorFlow
OpenCV (cv2)
numpy
matplotlib

# Data Requirements

Insurance data: CSV file with columns for age, sex, BMI, smoking status, region, and charges
Cancer diagnosis data: CSV file with patient information and cancer status
Image data: Organized in directories with benign and malignant FNA images

# Output
Each component generates:

Performance metrics
Visualization plots
Model evaluation results
Prediction outputs
