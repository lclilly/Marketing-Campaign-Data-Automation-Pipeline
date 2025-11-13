#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Marketing Campaign Complete Automation Pipeline
================================================
Purpose: All-in-one script for data cleaning and automation
Author: Lilian
Date: November 2024

This combines the cleaner and automation into a single file for simplicity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DATA CLEANER (From Day 2)
# ============================================================================

class MarketingDataCleaner:
    """
    A comprehensive data cleaning pipeline for marketing campaign data.
    """
    
    def __init__(self, filepath):
        """Initialize with the messy data file."""
        self.filepath = filepath
        self.df_raw = None
        self.df_clean = None
        self.cleaning_report = {
            'total_records_initial': 0,
            'duplicates_removed': 0,
            'dates_standardized': 0,
            'missing_values_found': 0,
            'missing_values_handled': 0,
            'anomalies_detected': 0,
            'text_fields_cleaned': 0,
            'currency_conversions': 0,
            'total_records_final': 0
        }
    
    def load_data(self):
        """Load the raw data from CSV."""
        print("📂 Loading data...")
        self.df_raw = pd.read_csv(self.filepath)
        self.df_clean = self.df_raw.copy()
        self.cleaning_report['total_records_initial'] = len(self.df_clean)
        print(f"   ✓ Loaded {len(self.df_clean)} records")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows based on key columns."""
        print("\n🔍 Removing duplicates...")
        initial_count = len(self.df_clean)
        
        self.df_clean = self.df_clean.drop_duplicates(
            subset=['date', 'campaign_name', 'channel', 'impressions', 'clicks'],
            keep='first'
        )
        
        duplicates_removed = initial_count - len(self.df_clean)
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        print(f"   ✓ Removed {duplicates_removed} duplicate records")
        return self
    
    def standardize_dates(self):
        """Convert all date formats to standard YYYY-MM-DD format."""
        print("\n📅 Standardizing date formats...")
        
        def parse_date(date_str):
            """Try multiple date formats to parse the date."""
            if pd.isna(date_str):
                return pd.NaT
            
            if isinstance(date_str, datetime):
                return date_str
            
            formats = [
                '%m/%d/%Y', '%Y-%m-%d', '%d-%b-%Y', '%B %d, %Y',
                '%d/%m/%Y', '%Y/%m/%d',
            ]
            
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            
            try:
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        dates_before = self.df_clean['date'].apply(lambda x: isinstance(x, str)).sum()
        self.df_clean['date'] = self.df_clean['date'].apply(parse_date)
        self.cleaning_report['dates_standardized'] = dates_before
        
        print(f"   ✓ Standardized {dates_before} date entries")
        print(f"   ✓ Date range: {self.df_clean['date'].min().date()} to {self.df_clean['date'].max().date()}")
        return self
    
    def clean_text_fields(self):
        """Clean and standardize text fields."""
        print("\n🧹 Cleaning text fields...")
        
        text_columns = ['campaign_name', 'channel']
        changes = 0
        
        for col in text_columns:
            original = self.df_clean[col].copy()
            
            self.df_clean[col] = self.df_clean[col].str.strip()
            
            if col == 'campaign_name':
                self.df_clean[col] = self.df_clean[col].str.replace('_', ' ')
                self.df_clean[col] = self.df_clean[col].str.replace(r'\s+', ' ', regex=True)
                self.df_clean[col] = self.df_clean[col].str.title()
            else:
                channel_mapping = {
                    'google ads': 'Google Ads',
                    'facebook': 'Facebook',
                    'email': 'Email',
                    'instagram': 'Instagram',
                    'linkedin': 'LinkedIn'
                }
                self.df_clean[col] = self.df_clean[col].str.lower().map(
                    lambda x: channel_mapping.get(x, x.title() if isinstance(x, str) else x)
                )
            
            changes += (original != self.df_clean[col]).sum()
        
        self.cleaning_report['text_fields_cleaned'] = changes
        print(f"   ✓ Cleaned {changes} text field entries")
        return self
    
    def convert_currency(self):
        """Convert currency strings to numeric values."""
        print("\n💰 Converting currency formats...")
        
        def clean_currency(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, (int, float)):
                return float(value)
            
            cleaned = str(value).replace('$', '').replace(',', '').strip()
            try:
                return float(cleaned)
            except:
                return np.nan
        
        currency_columns = ['spend', 'revenue']
        conversions = 0
        
        for col in currency_columns:
            original = self.df_clean[col].copy()
            self.df_clean[col] = self.df_clean[col].apply(clean_currency)
            conversions += original.apply(lambda x: isinstance(x, str)).sum()
        
        self.cleaning_report['currency_conversions'] = conversions
        print(f"   ✓ Converted {conversions} currency values to numeric")
        return self
    
    def handle_missing_values(self):
        """Intelligently handle missing values."""
        print("\n🔧 Handling missing values...")
        
        missing_before = self.df_clean.isnull().sum().sum()
        self.cleaning_report['missing_values_found'] = missing_before
        
        self.df_clean['missing_data_flag'] = self.df_clean[
            ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        ].isnull().any(axis=1)
        
        mask = self.df_clean['conversions'].isnull() & self.df_clean['clicks'].notnull()
        self.df_clean.loc[mask, 'conversions'] = 0
        
        missing_after = self.df_clean[
            ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        ].isnull().sum().sum()
        
        self.cleaning_report['missing_values_handled'] = missing_before - missing_after
        
        print(f"   ✓ Found {missing_before} missing values")
        print(f"   ✓ Handled {missing_before - missing_after} missing values")
        return self
    
    def detect_anomalies(self):
        """Detect and flag anomalous data points."""
        print("\n🚨 Detecting anomalies...")
        
        impossible_clicks = (self.df_clean['clicks'] > self.df_clean['impressions']) & \
                           (self.df_clean['clicks'].notnull()) & \
                           (self.df_clean['impressions'].notnull())
        
        impossible_conversions = (self.df_clean['conversions'] > self.df_clean['clicks']) & \
                                (self.df_clean['conversions'].notnull()) & \
                                (self.df_clean['clicks'].notnull())
        
        Q1_spend = self.df_clean['spend'].quantile(0.25)
        Q3_spend = self.df_clean['spend'].quantile(0.75)
        IQR_spend = Q3_spend - Q1_spend
        spend_outliers = (self.df_clean['spend'] > (Q3_spend + 3 * IQR_spend)) | \
                        (self.df_clean['spend'] < (Q1_spend - 3 * IQR_spend))
        
        self.df_clean['anomaly_impossible_clicks'] = impossible_clicks
        self.df_clean['anomaly_impossible_conversions'] = impossible_conversions
        self.df_clean['anomaly_spend_outlier'] = spend_outliers
        
        self.df_clean['has_anomaly'] = (
            self.df_clean['anomaly_impossible_clicks'] | 
            self.df_clean['anomaly_impossible_conversions'] | 
            self.df_clean['anomaly_spend_outlier']
        )
        
        total_anomalies = self.df_clean['has_anomaly'].sum()
        self.cleaning_report['anomalies_detected'] = total_anomalies
        
        print(f"   ✓ Detected {impossible_clicks.sum()} records with clicks > impressions")
        print(f"   ✓ Detected {impossible_conversions.sum()} records with conversions > clicks")
        print(f"   ✓ Detected {spend_outliers.sum()} spend outliers")
        return self
    
    def calculate_metrics(self):
        """Calculate derived marketing metrics."""
        print("\n📊 Calculating marketing metrics...")
        
        self.df_clean['ctr'] = np.where(
            self.df_clean['impressions'] > 0,
            (self.df_clean['clicks'] / self.df_clean['impressions']) * 100,
            np.nan
        )
        
        self.df_clean['conversion_rate'] = np.where(
            self.df_clean['clicks'] > 0,
            (self.df_clean['conversions'] / self.df_clean['clicks']) * 100,
            np.nan
        )
        
        self.df_clean['cpc'] = np.where(
            self.df_clean['clicks'] > 0,
            self.df_clean['spend'] / self.df_clean['clicks'],
            np.nan
        )
        
        self.df_clean['cpa'] = np.where(
            self.df_clean['conversions'] > 0,
            self.df_clean['spend'] / self.df_clean['conversions'],
            np.nan
        )
        
        self.df_clean['roas'] = np.where(
            self.df_clean['spend'] > 0,
            self.df_clean['revenue'] / self.df_clean['spend'],
            np.nan
        )
        
        self.df_clean['profit'] = self.df_clean['revenue'] - self.df_clean['spend']
        
        print(f"   ✓ Calculated 6 key metrics")
        return self
    
    def add_data_quality_score(self):
        """Add an overall data quality score for each record."""
        print("\n⭐ Adding data quality scores...")
        
        quality_score = 100
        
        self.df_clean['quality_score'] = quality_score
        self.df_clean.loc[self.df_clean['missing_data_flag'], 'quality_score'] -= 20
        self.df_clean.loc[self.df_clean['has_anomaly'], 'quality_score'] -= 30
        
        self.df_clean['quality_tier'] = pd.cut(
            self.df_clean['quality_score'],
            bins=[0, 50, 80, 100],
            labels=['Poor', 'Fair', 'Good']
        )
        
        print(f"   ✓ Quality distribution:")
        print(f"      Good: {(self.df_clean['quality_tier'] == 'Good').sum()} records")
        print(f"      Fair: {(self.df_clean['quality_tier'] == 'Fair').sum()} records")
        print(f"      Poor: {(self.df_clean['quality_tier'] == 'Poor').sum()} records")
        return self
    
    def save_clean_data(self, output_path='clean_marketing_data.csv'):
        """Save the cleaned data to CSV."""
        print(f"\n💾 Saving cleaned data...")
        
        core_columns = ['campaign_id', 'date', 'campaign_name', 'channel']
        metric_columns = ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        calculated_columns = ['ctr', 'conversion_rate', 'cpc', 'cpa', 'roas', 'profit']
        quality_columns = ['quality_score', 'quality_tier', 'missing_data_flag', 'has_anomaly']
        anomaly_columns = ['anomaly_impossible_clicks', 'anomaly_impossible_conversions', 
                          'anomaly_spend_outlier']
        
        final_columns = (core_columns + metric_columns + calculated_columns + 
                        quality_columns + anomaly_columns)
        
        self.df_clean = self.df_clean[final_columns]
        self.df_clean.to_csv(output_path, index=False)
        
        self.cleaning_report['total_records_final'] = len(self.df_clean)
        
        print(f"   ✓ Saved {len(self.df_clean)} cleaned records")
        return self
    
    def generate_report(self):
        """Generate a comprehensive cleaning report."""
        print("\n" + "="*60)
        print("📋 DATA CLEANING REPORT")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"   Initial records:        {self.cleaning_report['total_records_initial']}")
        print(f"   Duplicates removed:     {self.cleaning_report['duplicates_removed']}")
        print(f"   Final records:          {self.cleaning_report['total_records_final']}")
        
        print(f"\n🔧 Cleaning Actions:")
        print(f"   Dates standardized:     {self.cleaning_report['dates_standardized']}")
        print(f"   Text fields cleaned:    {self.cleaning_report['text_fields_cleaned']}")
        print(f"   Currency conversions:   {self.cleaning_report['currency_conversions']}")
        print(f"   Missing values handled: {self.cleaning_report['missing_values_handled']}")
        print(f"   Anomalies detected:     {self.cleaning_report['anomalies_detected']}")
        
        print(f"\n⏱️  Estimated Time Saved:")
        print(f"   Manual process:         ~4 hours/week")
        print(f"   Automated process:      ~5 minutes/week")
        print(f"   Annual time saved:      ~208 hours")
        
        print("\n" + "="*60)
        print("✅ CLEANING COMPLETE!")
        print("="*60 + "\n")
        
        return self.cleaning_report
    
    def run_full_pipeline(self, output_path='clean_marketing_data.csv'):
        """Execute the complete cleaning pipeline."""
        print("🚀 Starting Marketing Data Cleaning Pipeline")
        print("="*60 + "\n")
        
        (self
         .load_data()
         .remove_duplicates()
         .standardize_dates()
         .clean_text_fields()
         .convert_currency()
         .handle_missing_values()
         .detect_anomalies()
         .calculate_metrics()
         .add_data_quality_score()
         .save_clean_data(output_path)
         .generate_report())
        
        return self.df_clean


# ============================================================================
# PART 2: AUTOMATION & INSIGHTS (From Day 3)
# ============================================================================

class MarketingAutomation:
    """
    Automated pipeline for marketing data processing and insights.
    """
    
    def __init__(self, config_file='config.json'):
        """Initialize with configuration."""
        self.config = self.load_config(config_file)
        self.last_run_time = None
    
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        default_config = {
            'input_file': 'messy_marketing_data.csv',
            'output_file': 'clean_marketing_data.csv'
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def clean_data(self):
        """Run the data cleaning pipeline."""
        print(f"\n{'='*60}")
        print(f"🤖 AUTOMATED CLEANING: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        try:
            cleaner = MarketingDataCleaner(self.config['input_file'])
            clean_df = cleaner.run_full_pipeline(self.config['output_file'])
            
            self.last_run_time = datetime.now()
            self.clean_data_df = clean_df
            self.cleaning_report = cleaner.cleaning_report
            
            return clean_df
            
        except Exception as e:
            print(f"❌ Error during cleaning: {str(e)}")
            raise
    
    def generate_insights(self, df):
        """Generate actionable business insights."""
        print(f"\n💡 Generating Business Insights...")
        
        insights = []
        
        # 1. Best performing channel
        channel_performance = df.groupby('channel').agg({
            'roas': 'mean',
            'spend': 'sum',
            'revenue': 'sum'
        }).round(2)
        
        best_channel = channel_performance['roas'].idxmax()
        best_roas = channel_performance['roas'].max()
        
        insights.append(
            f"📊 {best_channel} has the highest ROAS at {best_roas:.2f}x - "
            f"consider shifting more budget here"
        )
        
        # 2. Underperforming campaigns
        low_performers = df[df['roas'] < 1.0]
        if len(low_performers) > 0:
            total_wasted = low_performers['spend'].sum() - low_performers['revenue'].sum()
            insights.append(
                f"⚠️ {len(low_performers)} campaigns have ROAS below 1.0x - "
                f"potential savings of ${total_wasted:,.2f} by pausing them"
            )
        
        # 3. Day of week analysis
        df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
        dow_performance = df.groupby('day_of_week')['roas'].mean().round(2)
        best_day = dow_performance.idxmax()
        
        insights.append(
            f"📅 {best_day} shows the best average ROAS - "
            f"schedule high-value campaigns on this day"
        )
        
        # 4. Campaign efficiency
        efficient_campaigns = df.nsmallest(3, 'cpa')[['campaign_name', 'cpa', 'conversions']]
        if len(efficient_campaigns) > 0:
            top_campaign = efficient_campaigns.iloc[0]
            insights.append(
                f"🎯 '{top_campaign['campaign_name']}' has the lowest CPA at ${top_campaign['cpa']:.2f} - "
                f"model other campaigns after this one"
            )
        
        # Print insights
        print("\n" + "="*60)
        print("📈 KEY BUSINESS INSIGHTS")
        print("="*60)
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight}")
        print("\n" + "="*60)
        
        return insights
    
    def run_pipeline(self):
        """Execute the complete automation pipeline."""
        try:
            # Clean data
            clean_df = self.clean_data()
            
            # Generate insights
            insights = self.generate_insights(clean_df)
            
            print(f"\n{'='*60}")
            print(f"✅ AUTOMATION COMPLETE")
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'timestamp': self.last_run_time,
                'records_processed': len(clean_df),
                'insights': insights
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🤖 Marketing Campaign Automation Pipeline")
    print("="*60 + "\n")
    
    # Create default config if it doesn't exist
    if not os.path.exists('config.json'):
        config = {
            'input_file': 'messy_marketing_data.csv',
            'output_file': 'clean_marketing_data.csv'
        }
        with open('config.json', 'w') as f:
            json.dump(config, indent=4, fp=f)
        print("✅ Created config.json\n")
    
    # Run automation
    automation = MarketingAutomation('config.json')
    result = automation.run_pipeline()
    
    if result['success']:
        print("\n💡 Next Steps:")
        print("   1. Review clean_marketing_data.csv")
        print("   2. Screenshot the console output for your portfolio")
        print("   3. Move to Day 4: Build your dashboard!")
    else:
        print("\n⚠️ Check the error message above and try again")


# In[ ]:




