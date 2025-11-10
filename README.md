# Heart Disease Prediction - Advanced ML Application

A comprehensive machine learning application for heart disease prediction with advanced techniques including SMOTE, XGBoost, LightGBM, hyperparameter tuning, and AOP-based logging.

## 🚀 Features

### Advanced ML Techniques
- **Class Imbalance Handling**: SMOTE, ADASYN, SMOTE+Tomek, Random Undersampling
- **Advanced Models**: XGBoost, LightGBM, Random Forest, Gradient Boosting, SVM, Logistic Regression
- **Hyperparameter Tuning**: RandomizedSearchCV for efficient parameter optimization
- **Feature Selection**: RFE, Statistical tests, Correlation analysis
- **Cross-Validation**: 5-fold stratified cross-validation with multiple metrics
- **Performance Validation**: Automated threshold-based validation

### Architecture & Design Patterns
- **Service-Oriented Architecture**: Modular services for data, EDA, and model operations
- **AOP-Based Logging**: Aspect-oriented programming for comprehensive logging
- **Error Handling**: Centralized error handling with custom exceptions
- **Configuration Management**: Environment-based configuration with .env files
- **Data Validation**: Comprehensive data validation and quality checks

### Visualization & EDA
- **Comprehensive EDA**: Target distribution, correlation analysis, feature analysis
- **Risk Factor Analysis**: Detailed risk factor visualization
- **Model Comparison**: Performance comparison across multiple algorithms
- **Feature Importance**: Feature importance analysis and visualization
- **ROC Curves**: ROC curve analysis with AUC scores

## 📁 Project Structure

```
Heart-Disease/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Main application entry point
│   ├── config.py              # Configuration management
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_service.py    # Data processing service
│   │   ├── eda_service.py     # EDA and visualization service
│   │   └── model_service.py   # Model training service
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py   # AOP-based logging
│       └── error_handling.py   # Error handling and validation
├── config.env                 # Environment configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── Heart_Disease_Prediction.ipynb  # Original Jupyter notebook
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Heart-Disease
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Edit `config.env` to customize settings
   - Adjust paths, thresholds, and model parameters as needed

## 🚀 Usage

### Run Complete Pipeline
```bash
python app/main.py
```

### Configuration Options
Edit `config.env` to customize:
- Data sources and paths
- Model parameters
- Performance thresholds
- Logging settings
- Visualization settings

## 📊 Services Overview

### DataService
- Data loading and preprocessing
- Feature engineering
- Categorical encoding
- Feature scaling
- Class imbalance handling
- Feature selection
- Data validation

### EDAService
- Comprehensive data overview
- Target distribution analysis
- Correlation analysis
- Feature analysis
- Risk factor analysis
- Model comparison visualization
- ROC curve analysis

### ModelService
- Model initialization and training
- Hyperparameter tuning
- Cross-validation evaluation
- Model selection
- Performance validation
- Model persistence

## 🔧 Configuration

### Data Configuration
```env
DATA_URL=https://raw.githubusercontent.com/kushal140/datasets/main/heart_2020_cleaned.csv
DATASET_PATH=./data/heart_disease.csv
OUTPUT_DIR=./outputs
MODEL_DIR=./models
```

### Model Configuration
```env
TEST_SIZE=0.2
RANDOM_STATE=42
CV_FOLDS=5
N_ITER_RANDOM_SEARCH=20
```

### Performance Thresholds
```env
MIN_ACCURACY_THRESHOLD=0.7
MIN_F1_THRESHOLD=0.6
MIN_AUC_THRESHOLD=0.7
```

## 📈 Outputs

The application generates:
- **Models**: Best model and all trained models
- **Preprocessing Artifacts**: Scalers, encoders, SMOTE
- **Visualizations**: EDA plots, model comparisons, ROC curves
- **Reports**: Comprehensive summary reports
- **Logs**: Detailed execution logs

## 🔍 Advanced Features

### AOP-Based Logging
- Function call logging
- Performance monitoring
- Data information logging
- Model-specific logging

### Error Handling
- Custom exceptions
- Data validation
- Model validation
- File operation handling

### Performance Monitoring
- Execution time tracking
- Performance threshold validation
- Cross-validation analysis
- Model comparison metrics

## 🎯 Key Improvements Over Notebook

1. **Modular Architecture**: Service-oriented design for maintainability
2. **Advanced ML Techniques**: SMOTE, XGBoost, LightGBM, hyperparameter tuning
3. **Comprehensive Logging**: AOP-based logging for debugging and monitoring
4. **Error Handling**: Robust error handling and validation
5. **Configuration Management**: Environment-based configuration
6. **Performance Validation**: Automated performance threshold validation
7. **Visualization**: Comprehensive visualization suite
8. **Documentation**: Detailed documentation and code comments

## 📝 Logging

The application provides comprehensive logging:
- **Function calls**: Track all function executions
- **Performance metrics**: Monitor execution times
- **Data information**: Log data shapes and distributions
- **Model results**: Log training and evaluation results
- **Error tracking**: Detailed error logging and handling

## 🔄 Pipeline Steps

1. **Data Loading**: Load and validate dataset
2. **EDA**: Generate comprehensive exploratory analysis
3. **Preprocessing**: Feature engineering and encoding
4. **Scaling**: Feature scaling and normalization
5. **Class Balancing**: Handle class imbalance with SMOTE
6. **Feature Selection**: Select best features using RFE
7. **Data Splitting**: Split into train/test sets
8. **Model Training**: Train multiple algorithms
9. **Hyperparameter Tuning**: Optimize model parameters
10. **Cross-Validation**: Evaluate with k-fold CV
11. **Model Selection**: Select best performing model
12. **Final Evaluation**: Comprehensive model evaluation
13. **Visualization**: Generate all plots and charts
14. **Persistence**: Save models and artifacts
15. **Reporting**: Generate summary reports

## 🎉 Results

The application provides:
- **Best Model**: Automatically selected based on performance
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Feature Importance**: Most important features for prediction
- **Visualizations**: Comprehensive plots and charts
- **Validation**: Performance threshold validation
- **Artifacts**: All models and preprocessing components saved

## 📞 Support

For issues or questions:
1. Check the logs in `./logs/heart_disease_prediction.log`
2. Review the configuration in `config.env`
3. Examine the output files in `./outputs/`
4. Check the saved models in `./models/`