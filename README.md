# Heart Attack Risk Prediction

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Contact](#contact)

## About the Project
This project aims to predict the presence of heart disease in patients using machine learning models. Various classification algorithms are applied to analyze patient data and determine the likelihood of heart disease.

## Features
- Predicts heart disease based on multiple health parameters
- Supports multiple machine learning models
- Provides accuracy comparison for different models
- User-friendly interface for making predictions

## Dataset
The dataset used in this project contains health-related attributes such as age, cholesterol levels, blood pressure, and more. The data is preprocessed to handle missing values and improve model performance.

## Installation
To run this project locally, follow these steps:

```sh
# Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git

# Navigate to the project directory
cd heart-disease-prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## Usage
Run the model to make predictions:

```sh
python app.py
```

Or if using a Jupyter Notebook, open `Heart_Disease_Prediction.ipynb` and execute the cells.

## Models Implemented
The following machine learning models were implemented and evaluated:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest** 
- **Decision Tree**
- **Naive Bayes**
- **SVM**

## Results
The accuracy of different models on the dataset:
| Model                | Accuracy  |
|----------------------|-----------|
| K-Nearest Neighbors | 80.02%     |
| Logistic Regression | 82.23%     |
| Random Forest       | 80.2%      |
| Decision Tree       | 79.7%      |
| Naive Bayes         | 82.23%     |
| SVM                 | 83.25%     |

## Technologies Used
- **Python**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn** (for visualization)

## Contributing
Contributions are welcome! If you would like to improve this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added a new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## Contact
📧 Email: cmkarthik.rallabhandi@example.com  
🔗 GitHub: [rckarthik3001](https://github.com/rckarthik3001)

