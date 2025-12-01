

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

from app.config import config
from app.utils.logging_utils import aop_logger
from app.utils.error_handling import error_handler


class EDAService:
    """Service for Exploratory Data Analysis and Visualization"""
    
    def __init__(self):
        self.logger = aop_logger.logger
        self._setup_plotting()
    
    def _setup_plotting(self) -> None:
        """Setup plotting configuration"""
        plt.style.use(config.visualization.STYLE)
        sns.set_palette("husl")
        self.logger.info("Plotting configuration initialized")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def generate_data_overview(self, df: pd.DataFrame, output_dir: str) -> Dict:
        """Generate comprehensive data overview"""
        self.logger.info("Generating data overview")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Basic statistics
        overview = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'target_distribution': df['HeartDisease'].value_counts().to_dict() if 'HeartDisease' in df.columns else None
        }
        
        # Save overview to file
        overview_file = os.path.join(output_dir, 'data_overview.txt')
        with open(overview_file, 'w') as f:
            f.write("=== DATA OVERVIEW ===\n")
            f.write(f"Dataset shape: {df.shape}\n")
            f.write(f"Memory usage: {overview['memory_usage']:.2f} MB\n")
            f.write(f"Data types: {overview['dtypes']}\n")
            f.write(f"Missing values: {overview['missing_values']}\n")
            if overview['target_distribution']:
                f.write(f"Target distribution: {overview['target_distribution']}\n")
        
        self.logger.info(f"Data overview saved to {overview_file}")
        return overview
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_target_distribution(self, df: pd.DataFrame, output_dir: str) -> None:
        """Plot target variable distribution"""
        self.logger.info("Creating target distribution plots")
        
        fig, axes = plt.subplots(2, 2, figsize=config.visualization.figure_size_large)
        
        # Heart Disease Distribution
        heart_disease_counts = df['HeartDisease'].value_counts()
        axes[0,0].pie(heart_disease_counts.values, labels=['No Heart Disease', 'Heart Disease'], 
                      autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        axes[0,0].set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
        
        # BMI Distribution by Heart Disease
        sns.histplot(data=df, x='BMI', hue='HeartDisease', kde=True, ax=axes[0,1])
        axes[0,1].set_title('BMI Distribution by Heart Disease Status')
        axes[0,1].set_xlabel('BMI')
        
        # Age Category vs Heart Disease
        age_heart = pd.crosstab(df['AgeCategory'], df['HeartDisease'])
        age_heart_pct = age_heart.div(age_heart.sum(axis=1), axis=0)
        age_heart_pct.plot(kind='bar', ax=axes[1,0], stacked=True)
        axes[1,0].set_title('Heart Disease Rate by Age Category')
        axes[1,0].set_xlabel('Age Category')
        axes[1,0].set_ylabel('Proportion')
        axes[1,0].legend(['No Heart Disease', 'Heart Disease'])
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Physical Health vs Heart Disease
        sns.boxplot(data=df, x='HeartDisease', y='PhysicalHealth', ax=axes[1,1])
        axes[1,1].set_title('Physical Health Score by Heart Disease Status')
        axes[1,1].set_xlabel('Heart Disease')
        axes[1,1].set_ylabel('Physical Health Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Target distribution plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_correlation_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """Plot correlation analysis"""
        self.logger.info("Creating correlation analysis plots")
        
        # Create correlation matrix for numerical variables
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=config.visualization.figure_size_large)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Correlation analysis plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_feature_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """Plot feature analysis"""
        self.logger.info("Creating feature analysis plots")
        
        categorical_features = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
                              'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes) and feature in df.columns:
                # Create crosstab
                crosstab = pd.crosstab(df[feature], df['HeartDisease'], normalize='index')
                crosstab.plot(kind='bar', ax=axes[i], stacked=True)
                axes[i].set_title(f'{feature} vs Heart Disease')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Proportion')
                axes[i].legend(['No Heart Disease', 'Heart Disease'])
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(categorical_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Feature analysis plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_risk_factor_analysis(self, df: pd.DataFrame, output_dir: str) -> None:
        """Plot risk factor analysis"""
        self.logger.info("Creating risk factor analysis plots")
        
        risk_factors = ['Smoking', 'AlcoholDrinking', 'Stroke', 'Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer']
        
        # Create risk factor analysis
        risk_analysis = {}
        for factor in risk_factors:
            if factor in df.columns:
                risk_rate = df.groupby(factor)['HeartDisease'].mean()
                risk_analysis[factor] = risk_rate.to_dict()
        
        # Plot risk factors
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, (factor, rates) in enumerate(risk_analysis.items()):
            if i < len(axes):
                values = list(rates.keys())
                rates_list = list(rates.values())
                
                axes[i].bar(values, rates_list, color=['lightblue', 'lightcoral'])
                axes[i].set_title(f'{factor} Risk Analysis')
                axes[i].set_xlabel(factor)
                axes[i].set_ylabel('Heart Disease Rate')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(risk_analysis), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_factor_analysis.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Risk factor analysis plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_class_balance_comparison(self, balancing_methods: Dict[str, Tuple], output_dir: str) -> None:
        """Plot class balance comparison"""
        self.logger.info("Creating class balance comparison plots")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (method_name, (X_bal, y_bal)) in enumerate(balancing_methods.items()):
            if i < len(axes):
                class_counts = np.bincount(y_bal)
                axes[i].pie(class_counts, labels=['No Heart Disease', 'Heart Disease'], 
                           autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
                axes[i].set_title(f'{method_name}\n({len(X_bal)} samples)')
        
        # Hide unused subplot
        if len(balancing_methods) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_balance_comparison.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Class balance comparison plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_feature_importance(self, feature_importance: pd.DataFrame, output_dir: str) -> None:
        """Plot feature importance"""
        self.logger.info("Creating feature importance plots")
        
        plt.figure(figsize=config.visualization.figure_size_medium)
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Feature importance plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_model_comparison(self, comparison_df: pd.DataFrame, output_dir: str) -> None:
        """Plot model comparison"""
        self.logger.info("Creating model comparison plots")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for i, metric in enumerate(metrics):
            if i < len(axes):
                sns.barplot(data=comparison_df, x=metric, y='Model', ax=axes[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_xlabel(metric)
        
        # Hide unused subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Model comparison plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_confusion_matrices(self, cm_balanced: np.ndarray, cm_original: np.ndarray, output_dir: str) -> None:
        """Plot confusion matrices"""
        self.logger.info("Creating confusion matrix plots")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Balanced test set confusion matrix
        sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix - Balanced Test Set')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Original test set confusion matrix
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Confusion Matrix - Original Test Set')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Confusion matrix plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def plot_roc_curves(self, fpr_balanced: np.ndarray, tpr_balanced: np.ndarray, 
                        fpr_original: np.ndarray, tpr_original: np.ndarray,
                        auc_balanced: float, auc_original: float, output_dir: str) -> None:
        """Plot ROC curves"""
        self.logger.info("Creating ROC curve plots")
        
        plt.figure(figsize=config.visualization.figure_size_medium)
        plt.plot(fpr_balanced, tpr_balanced, label=f'Balanced Test Set (AUC = {auc_balanced:.3f})')
        plt.plot(fpr_original, tpr_original, label=f'Original Test Set (AUC = {auc_original:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info("ROC curve plots saved")
    
    @error_handler.handle_data_errors
    @aop_logger.log_function_call
    def generate_comprehensive_report(self, df: pd.DataFrame, output_dir: str) -> None:
        """Generate comprehensive EDA report"""
        self.logger.info("Generating comprehensive EDA report")
        
        # Generate all plots
        self.plot_target_distribution(df, output_dir)
        self.plot_correlation_analysis(df, output_dir)
        self.plot_feature_analysis(df, output_dir)
        self.plot_risk_factor_analysis(df, output_dir)
        
        # Generate data overview
        overview = self.generate_data_overview(df, output_dir)
        
        self.logger.info("Comprehensive EDA report generated successfully")
