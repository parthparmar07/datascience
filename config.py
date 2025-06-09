"""
Project Configuration and Constants

Central configuration file for the Data Science Internship Portfolio.
Contains project settings, paths, and constants used across all tasks.

Author: Parth Parmar
Date: June 2025
"""

import os
from pathlib import Path

# Project Information
PROJECT_NAME = "Data Science Internship Portfolio"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Parth Parmar"
AUTHOR_EMAIL = "parth.parmar@example.com"
PROJECT_DESCRIPTION = """
A comprehensive data science portfolio showcasing machine learning expertise
across classification, regression, time series analysis, and data visualization.
"""

# Project Structure
PROJECT_ROOT = Path(__file__).parent
TASKS_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Task Configurations
TASK_CONFIGS = {
    'iris': {
        'name': 'Iris Flower Classification',
        'type': 'classification',
        'dataset': 'Iris.csv',
        'target_column': 'Species',
        'description': 'Multi-class classification of iris flower species',
        'algorithms': ['LogisticRegression', 'DecisionTree', 'RandomForest', 'SVM', 'KNN', 'NaiveBayes'],
        'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
        'visualization_types': ['correlation_matrix', 'pair_plot', 'confusion_matrix', 'feature_importance']
    },
    'unemployment': {
        'name': 'Unemployment Analysis in India',
        'type': 'analysis',
        'dataset': 'Unemployment in India.csv',
        'target_column': 'Estimated Unemployment Rate (%)',
        'description': 'Time series analysis of unemployment trends with COVID-19 impact assessment',
        'analysis_types': ['trend_analysis', 'regional_comparison', 'covid_impact', 'statistical_testing'],
        'visualization_types': ['time_series', 'choropleth', 'dashboard', 'comparative_analysis']
    },
    'carprice': {
        'name': 'Car Price Prediction',
        'type': 'regression',
        'dataset': 'car data.csv',
        'target_column': 'selling_price',
        'description': 'Regression analysis for car price prediction based on multiple features',
        'algorithms': ['LinearRegression', 'RandomForest', 'GradientBoosting', 'XGBoost'],
        'metrics': ['mse', 'rmse', 'mae', 'r2_score'],
        'visualization_types': ['scatter_plots', 'residual_plots', 'feature_importance', 'prediction_vs_actual']
    },
    'sales': {
        'name': 'Sales Prediction & Analytics',
        'type': 'regression',
        'dataset': 'Advertising.csv',
        'target_column': 'Sales',
        'description': 'Marketing analytics and sales prediction based on advertising spend',
        'algorithms': ['LinearRegression', 'PolynomialRegression', 'Ridge', 'Lasso'],
        'metrics': ['mse', 'rmse', 'mae', 'r2_score'],
        'features': ['TV', 'Radio', 'Newspaper'],
        'business_metrics': ['roi', 'cpa', 'roas'],
        'visualization_types': ['channel_analysis', 'roi_comparison', 'prediction_intervals']
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'scoring_metrics': {
        'classification': ['accuracy', 'precision', 'recall', 'f1'],
        'regression': ['neg_mean_squared_error', 'r2']
    },
    'hyperparameter_tuning': {
        'method': 'GridSearchCV',
        'cv': 5,
        'scoring': 'accuracy',  # Default, overridden per task
        'n_jobs': -1
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    'style': 'seaborn-v0_8',
    'palette': 'husl',
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12,
    'title_size': 16,
    'save_format': 'png',
    'save_quality': 95,
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#ffbb33',
        'danger': '#d62728',
        'info': '#17a2b8'
    }
}

# File Extensions and Formats
FILE_FORMATS = {
    'data': ['.csv', '.xlsx', '.json', '.parquet'],
    'images': ['.png', '.jpg', '.jpeg', '.svg'],
    'reports': ['.html', '.pdf', '.txt'],
    'code': ['.py', '.ipynb', '.R'],
    'config': ['.yaml', '.yml', '.json', '.ini']
}

# Performance Benchmarks
PERFORMANCE_BENCHMARKS = {
    'iris': {
        'accuracy_threshold': 0.95,
        'expected_accuracy': 0.98
    },
    'unemployment': {
        'insights_generated': 5,
        'visualizations_created': 4
    },
    'carprice': {
        'r2_threshold': 0.80,
        'expected_r2': 0.85
    },
    'sales': {
        'r2_threshold': 0.85,
        'expected_r2': 0.90
    }
}

# Data Quality Checks
DATA_QUALITY_CONFIG = {
    'max_missing_percentage': 20,
    'min_samples_per_class': 10,
    'outlier_detection_method': 'IQR',
    'correlation_threshold': 0.95,
    'variance_threshold': 0.01
}

# Output Configuration
OUTPUT_CONFIG = {
    'create_directories': True,
    'save_models': True,
    'save_plots': True,
    'save_reports': True,
    'model_format': 'joblib',
    'report_format': 'html'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_enabled': True,
    'console_enabled': True,
    'log_file': 'ds_portfolio.log'
}

# Environment Variables
ENV_CONFIG = {
    'python_version_min': '3.7',
    'required_packages': [
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0'
    ]
}

# API Configuration (for future extensions)
API_CONFIG = {
    'enable_api': False,
    'host': 'localhost',
    'port': 8000,
    'debug': True,
    'cors_enabled': True
}

# Documentation Configuration
DOCS_CONFIG = {
    'generate_docs': True,
    'doc_format': 'markdown',
    'include_code_examples': True,
    'include_visualizations': True,
    'api_documentation': False
}

def get_task_config(task_name):
    """
    Get configuration for a specific task
    
    Args:
        task_name (str): Name of the task
        
    Returns:
        dict: Task configuration
    """
    return TASK_CONFIGS.get(task_name, {})

def get_task_path(task_name):
    """
    Get the path for a specific task
    
    Args:
        task_name (str): Name of the task
        
    Returns:
        Path: Path to the task directory
    """
    task_mapping = {
        'iris': 'task1-iris',
        'unemployment': 'task2-unemployment',
        'carprice': 'task3-carprice',
        'sales': 'task4-sales'
    }
    
    return PROJECT_ROOT / task_mapping.get(task_name, task_name)

def create_output_directories():
    """Create necessary output directories if they don't exist"""
    directories = [OUTPUT_DIR, DOCS_DIR]
    
    for task_name in TASK_CONFIGS.keys():
        task_path = get_task_path(task_name)
        directories.extend([
            task_path / 'outputs',
            task_path / 'models',
            task_path / 'reports'
        ])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_environment():
    """
    Validate that the environment meets the requirements
    
    Returns:
        bool: True if environment is valid
    """
    import sys
    
    # Check Python version
    python_version = sys.version_info
    min_version = tuple(map(int, ENV_CONFIG['python_version_min'].split('.')))
    
    if python_version[:2] < min_version:
        print(f"❌ Python {ENV_CONFIG['python_version_min']}+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    
    # Check required packages
    missing_packages = []
    for package in ENV_CONFIG['required_packages']:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ Environment validation passed!")
    return True

# Initialize on import
if OUTPUT_CONFIG.get('create_directories', False):
    create_output_directories()
