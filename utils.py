"""
Data Science Utilities Module

Common utilities and helper functions used across all data science tasks
in the internship portfolio.

Author: Parth Parmar
Date: June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class DataScienceUtils:
    """
    Utility class containing common functions for data science tasks
    """
    
    @staticmethod
    def load_and_explore_data(file_path, display_info=True):
        """
        Load dataset and perform basic exploration
        
        Args:
            file_path (str): Path to the CSV file
            display_info (bool): Whether to display basic information
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            
            if display_info:
                print(f"📊 Dataset loaded successfully!")
                print(f"Shape: {df.shape}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                print("\n📋 Basic Information:")
                print(df.info())
                print("\n📈 Descriptive Statistics:")
                print(df.describe())
                print("\n🔍 Missing Values:")
                print(df.isnull().sum())
                
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    @staticmethod
    def plot_correlation_matrix(df, figsize=(12, 10), save_path=None):
        """
        Create and display correlation matrix heatmap
        
        Args:
            df (pd.DataFrame): Input dataframe
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("⚠️ No numeric columns found for correlation matrix")
            return
        
        plt.figure(figsize=figsize)
        correlation_matrix = numeric_df.corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Correlation matrix saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def evaluate_classification_model(y_true, y_pred, model_name="Model", verbose=True):
        """
        Comprehensive evaluation of classification models
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if verbose:
            print(f"\n📊 {model_name} Performance Metrics:")
            print("-" * 40)
            print(f"🎯 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"🎯 Precision: {metrics['precision']:.4f}")
            print(f"🎯 Recall:    {metrics['recall']:.4f}")
            print(f"🎯 F1-Score:  {metrics['f1_score']:.4f}")
            
        return metrics
    
    @staticmethod
    def evaluate_regression_model(y_true, y_pred, model_name="Model", verbose=True):
        """
        Comprehensive evaluation of regression models
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name (str): Name of the model
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        if verbose:
            print(f"\n📊 {model_name} Performance Metrics:")
            print("-" * 40)
            print(f"📈 MSE:  {metrics['mse']:.4f}")
            print(f"📈 RMSE: {metrics['rmse']:.4f}")
            print(f"📈 MAE:  {metrics['mae']:.4f}")
            print(f"📈 R²:   {metrics['r2']:.4f}")
            
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix with proper formatting
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
            
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_names, importance_values, 
                              title="Feature Importance", 
                              figsize=(10, 8), save_path=None):
        """
        Plot feature importance in a horizontal bar chart
        
        Args:
            feature_names: List of feature names
            importance_values: List of importance values
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=figsize)
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved to: {save_path}")
            
        plt.show()
    
    @staticmethod
    def compare_models(model_results, metric='accuracy', figsize=(12, 6), save_path=None):
        """
        Compare multiple models performance
        
        Args:
            model_results (dict): Dictionary with model names as keys and metrics as values
            metric (str): Metric to compare
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        models = list(model_results.keys())
        scores = [model_results[model][metric] for model in models]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Model Comparison - {metric.title()}', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric.title(), fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Model comparison plot saved to: {save_path}")
            
        plt.show()
    
    @staticmethod
    def data_preprocessing_summary(df, target_column=None):
        """
        Generate a comprehensive data preprocessing summary
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print("📋 Data Preprocessing Summary")
        print("=" * 40)
        print(f"📊 Total Rows: {summary['total_rows']:,}")
        print(f"📊 Total Columns: {summary['total_columns']}")
        print(f"🔢 Numeric Columns: {summary['numeric_columns']}")
        print(f"📝 Categorical Columns: {summary['categorical_columns']}")
        print(f"❓ Missing Values: {summary['missing_values']:,}")
        print(f"🔄 Duplicate Rows: {summary['duplicate_rows']:,}")
        print(f"💾 Memory Usage: {summary['memory_usage_mb']:.2f} MB")
        
        if target_column and target_column in df.columns:
            if df[target_column].dtype in ['object']:
                print(f"\n🎯 Target Variable Distribution:")
                print(df[target_column].value_counts())
            else:
                print(f"\n🎯 Target Variable Statistics:")
                print(df[target_column].describe())
        
        return summary

class ModelTracker:
    """
    Class to track and compare model performance across experiments
    """
    
    def __init__(self):
        self.experiments = []
    
    def add_experiment(self, model_name, metrics, dataset_info=None):
        """
        Add a new experiment result
        
        Args:
            model_name (str): Name of the model
            metrics (dict): Performance metrics
            dataset_info (dict): Information about the dataset used
        """
        experiment = {
            'model_name': model_name,
            'metrics': metrics,
            'dataset_info': dataset_info or {},
            'timestamp': pd.Timestamp.now()
        }
        self.experiments.append(experiment)
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on a specific metric
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            dict: Best experiment result
        """
        if not self.experiments:
            return None
        
        valid_experiments = [exp for exp in self.experiments 
                           if metric in exp['metrics']]
        
        if not valid_experiments:
            return None
        
        best_exp = max(valid_experiments, 
                      key=lambda x: x['metrics'][metric])
        return best_exp
    
    def get_summary_table(self):
        """
        Get a summary table of all experiments
        
        Returns:
            pd.DataFrame: Summary table
        """
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            row = {'Model': exp['model_name']}
            row.update(exp['metrics'])
            row['Timestamp'] = exp['timestamp']
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, file_path):
        """
        Export results to CSV file
        
        Args:
            file_path (str): Path to save the results
        """
        summary_df = self.get_summary_table()
        summary_df.to_csv(file_path, index=False)
        print(f"📊 Results exported to: {file_path}")

# Global instance for easy access
model_tracker = ModelTracker()
