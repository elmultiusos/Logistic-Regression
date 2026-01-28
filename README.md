# Heart Disease Prediction using Logistic Regression

## Exercise Summary

This project implements **logistic regression from scratch** for heart disease risk prediction. The implementation includes:

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the UCI Heart Disease Dataset
- **Model Training**: Gradient descent implementation with binary cross-entropy cost function
- **Decision Boundary Visualization**: 2D visualizations for multiple feature pairs
- **Regularization**: L2 (Ridge) regularization to prevent overfitting
- **Deployment Exploration**: Amazon SageMaker deployment preparation and documentation

## Dataset Description

### Source

**Kaggle Heart Disease Dataset**  
https://www.kaggle.com/datasets/neurocipher/heartdisease

### Statistics

- **Total Patients**: 303
- **Features**: 14 clinical measurements
- **Target**: Binary classification (Presence/Absence of heart disease)
- **Disease Presence Rate**: ~55% (moderately balanced)
- **Missing Values**: None
- **Age Range**: 29-77 years
- **Cholesterol Range**: 112-564 mg/dL

### Selected Features (8 for modeling)

1. **Age** - Patient age in years
2. **Cholesterol** - Serum cholesterol in mg/dL
3. **BP** - Resting blood pressure in mm Hg
4. **Max HR** - Maximum heart rate achieved during exercise
5. **ST Depression** - ST depression induced by exercise (ECG measurement)
6. **Number of Vessels Fluro** - Number of major vessels colored by fluoroscopy (0-3)
7. **Chest Pain Type** - Type of chest pain (1-4)
8. **Exercise Angina** - Exercise-induced angina (0=no, 1=yes)

## Model Performance

### Unregularized Logistic Regression (λ=0)

- **Training Accuracy**: 86.32%
- **Test Accuracy**: 85.71%
- **Test F1 Score**: 0.8571
- **Convergence**: 1000 iterations, learning rate α=0.01

### Regularized Logistic Regression (λ=0.01 - Optimal)

- **Training Accuracy**: 86.32%
- **Test Accuracy**: 86.81%
- **Test F1 Score**: 0.8696
- **Weight Norm Reduction**: 15.3% vs unregularized

### Feature Importance

Most influential features (by coefficient magnitude):

1. ST Depression (strongest predictor)
2. Number of Vessels Fluro
3. Exercise Angina
4. Max Heart Rate
5. Cholesterol

## Implementation Details

### Core Functions Implemented

```python
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def compute_cost(w, b, X, y):
    """Binary cross-entropy cost function"""
    # J(w,b) = -1/m * Σ[y*log(f) + (1-y)*log(1-f)]

def compute_gradient(w, b, X, y):
    """Compute gradients ∂J/∂w and ∂J/∂b"""
    # ∂J/∂w_j = 1/m * Σ[(f - y) * x_j]
    # ∂J/∂b = 1/m * Σ(f - y)

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """Optimize parameters using gradient descent"""
    # w := w - α * ∂J/∂w
    # b := b - α * ∂J/∂b
```

### Regularization

```python
def compute_cost_reg(w, b, X, y, lambda_):
    """Regularized cost with L2 penalty"""
    # J_reg = J + (λ/2m) * Σw_j²

def compute_gradient_reg(w, b, X, y, lambda_):
    """Regularized gradients"""
    # ∂J_reg/∂w_j = ∂J/∂w_j + (λ/m) * w_j
```

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib jupyter
```

### Execute the Notebook

```bash
jupyter notebook heart_disease_lr_analysis.ipynb
```

Or run all cells:

```bash
jupyter nbconvert --to notebook --execute heart_disease_lr_analysis.ipynb
```

## References

- **Dataset**: [Kaggle Heart Disease UCI](https://www.kaggle.com/datasets/neurocipher/heartdisease)
- **Theory**: Week 2 Classification notebooks (logistic regression fundamentals)
- **Deployment**: [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)

## License

This project is for educational purposes as part of a Machine Learning bootcamp assignment.

## Author

Juan Sebastian Buitrago Piñeros  
Universidad Escuela Colombiana de Ingenieria

## Submission

- **GitHub Repository**: [https://github.com/elmultiusos/Logistic-Regression]
- **Submission Date**: 28/01/2026
- **Course**: Machine Learning Bootcamp - Week 2
- **Assignment**: Heart Disease Prediction with Logistic Regression

---

**Note**: This implementation follows best practices by implementing algorithms from scratch for educational purposes. In production, optimized libraries like scikit-learn would typically be used, but this exercise demonstrates deep understanding of the underlying mathematics and algorithms.
