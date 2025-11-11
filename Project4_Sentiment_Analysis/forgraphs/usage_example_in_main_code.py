# Add this to your main.py after model training

def create_all_visualizations(df, models_results, best_model, X_test, y_test, vectorizer):
    """Create all visualizations for the project"""
    
    # 1. Data Exploration Visualizations
    print("Creating Data Exploration Visualizations...")
    visualizer = SentimentVisualizer(df)
    visualizer.create_comprehensive_dashboard()
    visualizer.create_wordclouds_grid()
    
    # 2. Model Performance Visualizations
    print("Creating Model Performance Visualizations...")
    model_viz = ModelPerformanceVisualizer()
    model_viz.plot_confusion_matrix_comparison(models_results, X_test, y_test)
    model_viz.plot_model_comparison_bar_chart(models_results)
    model_viz.plot_precision_recall_comparison(models_results, X_test, y_test)
    model_viz.create_performance_radar_chart(models_results, X_test, y_test)
    
    # 3. Interactive Visualizations
    print("Creating Interactive Visualizations...")
    interactive_viz = InteractiveVisualizations(df)
    interactive_viz.create_interactive_sentiment_dashboard()
    interactive_viz.create_interactive_word_cloud()
    
    # 4. Feature Importance Visualization
    print("Creating Feature Importance Visualization...")
    visualizer.plot_sentiment_heatmap(vectorizer, best_model)

# Call this function after model training in your main code
# create_all_visualizations(analyzer.df, analyzer.models, analyzer.best_model, 
#                          analyzer.X_test, analyzer.y_test, 
#                          analyzer.best_model.named_steps['vectorizer'])