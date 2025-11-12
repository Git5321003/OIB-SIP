import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_analysis_summary(analysis_results):
    """Print a summary of the analysis results"""
    
    stats = analysis_results['basic_stats']
    
    print("=" * 60)
    print("ANDROID APP MARKET ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Apps Analyzed: {stats['total_apps']:,}")
    print(f"Categories: {stats['total_categories']}")
    print(f"Genres: {stats['total_genres']}")
    print(f"Free Apps: {stats['free_apps']:,} ({stats['free_apps']/stats['total_apps']*100:.1f}%)")
    print(f"Paid Apps: {stats['paid_apps']:,} ({stats['paid_apps']/stats['total_apps']*100:.1f}%)")
    print(f"Average Rating: {stats['avg_rating']:.2f}/5.0")
    print(f"Average App Size: {stats['avg_size_mb']:.1f} MB")
    print(f"Total Installs: {stats['total_installs']:,}")
    print("\n" + "=" * 60)

def save_analysis_report(analysis_results, filename='reports/analysis_summary.txt'):
    """Save analysis results to a text file"""
    
    with open(filename, 'w') as f:
        f.write("ANDROID APP MARKET ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        stats = analysis_results['basic_stats']
        
        f.write("BASIC STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Apps: {stats['total_apps']:,}\n")
        f.write(f"Categories: {stats['total_categories']}\n")
        f.write(f"Free Apps: {stats['free_apps']:,}\n")
        f.write(f"Paid Apps: {stats['paid_apps']:,}\n")
        f.write(f"Average Rating: {stats['avg_rating']:.2f}\n")
        f.write(f"Total Installs: {stats['total_installs']:,}\n\n")
        
        # Top categories
        category_analysis = analysis_results['category_analysis']
        f.write("TOP CATEGORIES BY APP COUNT:\n")
        f.write("-" * 30 + "\n")
        for i, row in category_analysis.head(10).iterrows():
            f.write(f"{row['Category']}: {row['App_Count']} apps\n")
        
        f.write("\nPRICING INSIGHTS:\n")
        f.write("-" * 20 + "\n")
        pricing = analysis_results['pricing_analysis']
        f.write(f"Estimated Revenue from Paid Apps: ${pricing['revenue_estimate']:,.2f}\n")
        f.write(f"Average Price of Paid Apps: ${pricing['paid_apps_stats']['avg_price']:.2f}\n")

def format_large_number(num):
    """Format large numbers for better readability"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.0f}"
    