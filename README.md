# Application of Machine Learning Algorithms in Medical Image Classification
## Project Description
This project studys the application of machine learning algorithms in the field of medical image classification. Specifically, this project utilises two sub-datasets from the MedMNIST dataset including: BreastMNIST and BloodMNIST to complete two tasks: task A and task B. However, the project code only includes the results corresponding to the final selected hyperparameters and does not include the run results and visualisation plots produced by the hyperparameters tuning process.
* ***task A*** Classification of breast cancer images using Support Vector Machines (SVM). The purpose of the task is to identify benign and malignant breast cancer images and learn the hyperparameters selection and model training methods of SVM.
* ***task B*** Classification of blood cell images using Convolutional Neural Networks (CNN). The purpose of the task is to identify different blood cell images and learn the training process of CNN model.

## File Overview
The project is organized into the following folders and files:
- **A/** Contains the program code and functions required in Task A
  - `breastmnist_data_acquiring.py`: Acquires the breast cancer images for model training, validation and testing.
  - `breastmnist_data-processing.py`: Contains functions for analyzing and preprocessing the breast cancer images, as well as for SVM model training, validation, and testing.
  - `breastmnist_result_visualising.py`: Generates visualizations for data analysis and experimental results.

- **B/** Contains the program code and functions required in Task B
  - `bloodmnist_data_acquiring.py`: Acquires the blood cell images for model training, validation and testing.
  - `bloodmnist_data_processing.py`: Contains functions for preprocessing the blood cell images and constructing the CNN network architecture, as well as for model training, validation, and testing.
  - `bloodmnist_result_visualising.py`: Generates visualizations for the experimental results.

- **env/** Includes the descriptions of the environment required to run the programme
  - `environment.yml`: Defines the environment and its version
  - `requirements`: Lists python packages that required to run the code 

- **Datasets**: Stores the datasets that used in this project.
- **figures**: Stores the plots generated during the project, including images of model training and hyperparameters tuning process as well as the final results.
- **main.py**: The main script that contains the complete workflow code for the classification tasks of Task A and Task B.

## Required Packages
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `tqdm`
- `medmnist`


## How to Run the Code
1. **Open the terminal and use cd to navigate to the root directory**
2. **Create the Conda Environment:**
   ```bash
   sudo conda env create -f env/environment.yml
3. **Check the Environment:**
   ```bash
   conda info --envs
4. **Activate the Environment:**
   ```bash
   conda activate amls-final-project-env
5. **Install the required packages:**
   ```bash
   pip install -r env/requirements.txt
6. **Run the main script:**
   ```bash
   python main.py
## Purpose
The main purpose of this project is to learn how to apply machine learning algorithms to solve real-world problems, learn the process of model hyperparameter tuning and training, and understand the impact of hyperparameters on model performance.