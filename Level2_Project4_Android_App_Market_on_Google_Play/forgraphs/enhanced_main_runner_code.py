#!/usr/bin/env python3
"""
Main script to run the complete Android App Market analysis with enhanced visualizations
"""

import os
import sys
import pandas as pd
from src.data_cleaning import clean_dataset
from src.analysis_functions import perform_comprehensive_analysis
from src.visualization import create_all_visualizations, create_app_word_cloud
from src.utils import print_analysis_summary, save_analysis_report

def main():
    """Execute the complete analysis pipeline"""
    print("🚀 Starting Android App Market Analysis...")
    print("=" * 50)
    
    # Create directories
    directories = ['data', 'visualizations', 'reports', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories for visualizations
    viz_subdirs = ['category_distribution', 'rating_analysis', 'pricing_insights', 
                   'popularity_metrics', 'sentiment_analysis', 'comprehensive',
                   'size_analysis', 'content_rating', 'temporal_analysis']
    
    for subdir in viz_subdirs:
        os.makedirs(f'visualizations/{subdir}', exist_ok=True)
    
    try:
        # Data cleaning
        print("📊 Step 1: Data Cleaning...")
        df_cleaned = clean_dataset('data/apps.csv')
        df_cleaned.to_csv('data/apps_cleaned.csv', index=False)
        print(f"✅ Data cleaned: {len(df_cleaned)} apps")
        
        # Analysis
        print("\n📈 Step 2: Performing Analysis...")
        analysis_results = perform_comprehensive_analysis(df_cleaned)
        
        # Print summary
        print_analysis_summary(analysis_results)
        
        # Visualization
        print("\n🎨 Step 3: Creating Visualizations...")
        create_all_visualizations(df_cleaned, analysis_results)
        
        # Additional visualizations
        create_app_word_cloud(df_cleaned)
        
        # Save report
        print("\n📋 Step 4: Generating Reports...")
        save_analysis_report(analysis_results)
        
        print("\n" + "=" * 50)
        print("✅ Analysis Complete!")
        print("\n📁 Generated Files:")
        print("  - data/apps_cleaned.csv (Cleaned dataset)")
        print("  - visualizations/ (All graphs and dashboards)")
        print("  - reports/analysis_summary.txt (Analysis report)")
        print("\n🎯 Next Steps:")
        print("  - Open HTML files in visualizations/ for interactive dashboards")
        print("  - Check PNG files for static charts")
        print("  - Review analysis_summary.txt for key insights")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    