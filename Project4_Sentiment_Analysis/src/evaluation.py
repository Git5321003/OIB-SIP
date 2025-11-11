import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for the model
    """
    if hasattr(model.named_steps['classifier'], 'coef_'):
        # For linear models
        importance = model.named_steps['classifier'].coef_[0]
    elif hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # For tree-based models
        importance = model.named_steps['classifier'].feature_importances_
    else:
        print("Model doesn't support feature importance")
        return
    
    # Get feature names from vectorizer
    feature_names = model.named_steps['vectorizer'].get_feature_names_out()
    
    # Create dataframe
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_imp_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()
    
    return feature_imp_df

def analyze_misclassifications(model, X_test, y_test):
    """
    Analyze misclassified examples
    """
    y_pred = model.predict(X_test)
    misclassified = X_test[y_test != y_pred]
    actual_labels = y_test[y_test != y_pred]
    predicted_labels = y_pred[y_test != y_pred]
    
    misclassified_df = pd.DataFrame({
        'text': misclassified,
        'actual': actual_labels,
        'predicted': predicted_labels
    })
    
    print(f"Number of misclassified examples: {len(misclassified_df)}")
    print("\nSample misclassifications:")
    print(misclassified_df.head(10))
    
    return misclassified_df