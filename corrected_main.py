import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """
    Task 1: Data Loading and Cleaning
    """
    print("=" * 60)
    print("TASK 1: DATA LOADING AND CLEANING")
    print("=" * 60)
    
    df = pd.read_excel("test.xlsx")
    print(f"Original dataset shape: {df.shape}")
    
    df = df.dropna(subset=['body', 'date', 'from'])
    df = df[df['body'].str.strip() != '']
    print(f"After removing missing data: {df.shape}")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # type: ignore
    print(f"After date conversion: {df.shape}")
    
    df = df.drop_duplicates(subset=['from', 'date', 'body'])
    print(f"After removing duplicates: {df.shape}")
    
    def clean_message(text):
        text = str(text).lower()
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    df['cleaned_message'] = df['body'].apply(clean_message)
    df['word_count'] = df['cleaned_message'].str.split().str.len()
    df['char_count'] = df['cleaned_message'].str.len()
    
    print("Data cleaning completed!")
    return df

def perform_sentiment_labeling(df):
    """
    Task 1: Sentiment Labeling
    """
    print("\n" + "=" * 60)
    print("TASK 1: SENTIMENT LABELING")
    print("=" * 60)
    
    
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if not isinstance(text, str):
            return "Neutral"
        
        score = analyzer.polarity_scores(text)
        compound = score['compound']
        
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    
    df['sentiment_label'] = df['cleaned_message'].apply(get_sentiment)
    
    
    sentiment_score_map = {
        'Positive': 1,
        'Neutral': 0,
        'Negative': -1
    }
    df['sentiment_score'] = df['sentiment_label'].map(sentiment_score_map)
    
    print("Sentiment labeling completed!")
    print(f"Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
    
    return df

def exploratory_data_analysis(df):
    """
    Task 2: Exploratory Data Analysis (EDA)
    """
    print("\n" + "=" * 60)
    print("TASK 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    print("Dataset Information:")
    print(df.info())
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    sentiment_counts = df['sentiment_label'].value_counts()
    print(f"\nSentiment Distribution:\n{sentiment_counts}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.countplot(data=df, x='sentiment_label', ax=axes[0,0], palette='Set2')
    axes[0,0].set_title("Sentiment Label Distribution")
    axes[0,0].set_xlabel("Sentiment")
    axes[0,0].set_ylabel("Number of Messages")
    
    sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, 
                             colors=sns.color_palette("Set2"), ax=axes[0,1])
    axes[0,1].set_title("Sentiment Distribution")
    axes[0,1].set_ylabel("")
    
    df['month'] = df['date'].dt.to_period('M')
    monthly_trend = df.groupby(['month', 'sentiment_label']).size().unstack(fill_value=0)
    monthly_trend.plot(marker='o', ax=axes[1,0])
    axes[1,0].set_title("Monthly Sentiment Trend")
    axes[1,0].set_xlabel("Month")
    axes[1,0].set_ylabel("Message Count")
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True)
    axes[1,0].legend(title="Sentiment")
    
    sns.histplot(df['word_count'], bins=30, kde=True, color='purple', ax=axes[1,1])
    axes[1,1].set_title("Distribution of Word Counts in Messages")
    axes[1,1].set_xlabel("Word Count")
    axes[1,1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig('visualization/eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    top_senders = df['from'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_senders.index, x=top_senders.values, palette="coolwarm")
    plt.title("Top 10 Employees by Message Count")
    plt.xlabel("Messages Sent")
    plt.ylabel("Employee")
    plt.tight_layout()
    plt.savefig('visualization/top_employees.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("EDA completed and visualizations saved!")
    return df

def calculate_employee_scores(df):
    """
    Task 3: Employee Score Calculation
    """
    print("\n" + "=" * 60)
    print("TASK 3: EMPLOYEE SCORE CALCULATION")
    print("=" * 60)
    
    monthly_scores = df.groupby(['from', df['date'].dt.to_period('M')])['sentiment_score'].sum().reset_index()
    monthly_scores.columns = ['employee', 'month', 'monthly_sentiment_score']
    monthly_scores = monthly_scores.sort_values(['employee', 'month'])
    
    print(f"Monthly scores calculated for {len(monthly_scores)} employee-month combinations")
    print(f"Score range: {monthly_scores['monthly_sentiment_score'].min()} to {monthly_scores['monthly_sentiment_score'].max()}")
    
    monthly_scores.to_excel("monthly_employee_sentiment_scores.xlsx", index=False)
    print("Monthly scores saved to 'monthly_employee_sentiment_scores.xlsx'")
    
    return monthly_scores

def employee_ranking(monthly_scores):
    """
    Task 4: Employee Ranking
    """
    print("\n" + "=" * 60)
    print("TASK 4: EMPLOYEE RANKING")
    print("=" * 60)
    
    top_positive = (
        monthly_scores
        .sort_values(['month', 'monthly_sentiment_score', 'employee'], ascending=[True, False, True])
        .groupby('month')
        .head(3)
        .reset_index(drop=True)
    )
    top_positive['rank_type'] = 'Top Positive'
    
    top_negative = (
        monthly_scores
        .sort_values(['month', 'monthly_sentiment_score', 'employee'], ascending=[True, True, True])
        .groupby('month')
        .head(3)
        .reset_index(drop=True)
    )
    top_negative['rank_type'] = 'Top Negative'
    
    
    employee_ranking = pd.concat([top_positive, top_negative], ignore_index=True)
    employee_ranking = employee_ranking.sort_values(['month', 'rank_type', 'monthly_sentiment_score'], 
                                                   ascending=[True, True, False])
    
    print("Top 3 Positive and Negative Employees per Month:")
    print(employee_ranking.head(20))
    
    
    employee_ranking.to_excel("D:\\employee-sentiment-analysis\\employee_monthly_ranking.xlsx", index=False)
    print("Employee rankings saved to 'employee_monthly_ranking.xlsx'")
    
    return employee_ranking

def flight_risk_identification(df):
    """
    Task 5: Flight Risk Identification
    CORRECTED: According to project requirements - 4+ negative messages in a given month
    """
    print("\n" + "=" * 60)
    print("TASK 5: FLIGHT RISK IDENTIFICATION")
    print("=" * 60)
    
    neg_df = df[df['sentiment_label'] == 'Negative'].copy()
    neg_df['date'] = pd.to_datetime(neg_df['date'], errors='coerce')
    
    monthly_negative_counts = neg_df.groupby(['from', neg_df['date'].dt.to_period('M')]).size().reset_index(name='negative_count')
    
    flight_risk_employees = monthly_negative_counts[monthly_negative_counts['negative_count'] >= 4]['from'].unique()
    flight_risks = set(flight_risk_employees)
    
    print(f"Found {len(flight_risks)} employees flagged as flight risks")
    print("Flight Risk Employees:")
    for emp in flight_risks:
        print(f"- {emp}")
    
    

    flight_risk_df = pd.DataFrame({'employee': list(flight_risks)})
    flight_risk_df.to_excel("flight_risk_employees.xlsx", index=False)
    print("Flight risk list saved to 'flight_risk_employees.xlsx'")
    
    return flight_risk_df

def predictive_modeling(df):
    """
    Task 6: Enhanced Predictive Modeling with Advanced Features and Models
    """
    print("\n" + "=" * 60)
    print("TASK 6: ENHANCED PREDICTIVE MODELING")
    print("=" * 60)
    
    
    print("Creating enhanced features...")
    
    
    monthly_features = df.groupby(['from', df['date'].dt.to_period('M')]).agg({
        'cleaned_message': 'count',
        'word_count': ['sum', 'mean', 'std'],
        'char_count': ['sum', 'mean'],
        'sentiment_score': ['sum', 'mean', 'std'],
        'sentiment_label': lambda x: (x == 'Positive').sum(),
        'body': lambda x: (x == 'Negative').sum()
    }).reset_index()
    
   
    monthly_features.columns = ['employee', 'month', 'message_count', 'total_word_count', 
                               'avg_word_count', 'word_count_std', 'total_char_count', 
                               'avg_char_count', 'sentiment_score_sum', 'sentiment_score_mean', 
                               'sentiment_score_std', 'positive_count', 'negative_count']
    
   
    monthly_features['sentiment_ratio'] = monthly_features['sentiment_score_sum'] / monthly_features['message_count']
    monthly_features['positive_ratio'] = monthly_features['positive_count'] / monthly_features['message_count']
    monthly_features['negative_ratio'] = monthly_features['negative_count'] / monthly_features['message_count']
    monthly_features['word_per_message'] = monthly_features['total_word_count'] / monthly_features['message_count']
    monthly_features['char_per_word'] = monthly_features['total_char_count'] / monthly_features['total_word_count']
    
  
    monthly_features['month_num'] = monthly_features['month'].astype(str).str[:4].astype(int) * 12 + \
                                   monthly_features['month'].astype(str).str[5:7].astype(int)
    
   
    employee_stats = df.groupby('from').agg({
        'sentiment_score': ['mean', 'std'],
        'word_count': ['mean', 'std'],
        'cleaned_message': 'count'
    }).reset_index()
    employee_stats.columns = ['employee', 'emp_sentiment_mean', 'emp_sentiment_std', 
                             'emp_word_mean', 'emp_word_std', 'emp_total_messages']
    
    monthly_features = monthly_features.merge(employee_stats, on='employee', how='left')
    
    monthly_features['sentiment_volatility'] = monthly_features['sentiment_score_std'] / (monthly_features['emp_sentiment_std'] + 1e-6)
    monthly_features['word_volatility'] = monthly_features['word_count_std'] / (monthly_features['emp_word_std'] + 1e-6)
    monthly_features['message_intensity'] = monthly_features['message_count'] / monthly_features['emp_total_messages']
    
    monthly_features.dropna(inplace=True)
    
    print(f"Enhanced feature dataset shape: {monthly_features.shape}")
    
    feature_columns = ['message_count', 'total_word_count', 'avg_word_count', 'word_count_std',
                      'total_char_count', 'avg_char_count', 'sentiment_score_sum', 'sentiment_score_mean', 
                      'sentiment_score_std', 'positive_count', 'negative_count', 'sentiment_ratio',
                      'positive_ratio', 'negative_ratio', 'word_per_message', 'char_per_word',
                      'month_num', 'emp_sentiment_mean', 'emp_sentiment_std', 'emp_word_mean', 
                      'emp_word_std', 'emp_total_messages', 'sentiment_volatility', 'word_volatility', 
                      'message_intensity']
    
    X = monthly_features[feature_columns]
    y = monthly_features['sentiment_score_sum']  # Target: monthly sentiment score
    
    print(f"Number of features: {len(feature_columns)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    selector = SelectKBest(score_func=f_regression, k=15)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X_scaled.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features)  # type: ignore
    
    print(f"Selected top {len(selected_features)} features: {selected_features}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
   
    if xgb is not None:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {'r2': r2, 'rmse': rmse, 'model': model}
            
            print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    print(f"\nPerforming hyperparameter tuning for {best_name}...")
    
    if best_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif best_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        param_grid = {}
    
    if param_grid:
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_r2 = grid_search.best_score_
        print(f"Best parameters: {grid_search.best_params_}")
    
    if best_model is not None:
        y_pred_final = best_model.predict(X_test)
        final_r2 = r2_score(y_test, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    else:
        fallback_model = LinearRegression()
        fallback_model.fit(X_train, y_train)
        y_pred_final = fallback_model.predict(X_test)
        final_r2 = r2_score(y_test, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
        best_model = fallback_model
        best_name = "Linear Regression (Fallback)"
    
    print(f"\nFinal Model Performance ({best_name}):")
    print(f"R² Score: {final_r2:.3f}")
    print(f"RMSE: {final_rmse:.3f}")
    
    if best_model is not None and hasattr(best_model, 'feature_importances_') and not isinstance(best_model, LinearRegression):
        try:
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 10 Feature Importances:")
            print(importance_df.head(10))
            
            
            plt.figure(figsize=(10, 6))
            importance_df.head(10).plot(x='Feature', y='Importance', kind='barh')
            plt.title(f'Top 10 Feature Importances - {best_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('visualization/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
    else:
        print("\nFeature importance not available for this model type")
    
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
 
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    axes[0,0].bar(model_names, r2_scores, color='skyblue')
    axes[0,0].set_title('Model R² Score Comparison')
    axes[0,0].set_ylabel('R² Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
 
    axes[0,1].scatter(y_test, y_pred_final, alpha=0.6, color='green')
    y_test_min, y_test_max = float(y_test.min()), float(y_test.max())  # type: ignore
    axes[0,1].plot([y_test_min, y_test_max], [y_test_min, y_test_max], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Sentiment Score')
    axes[0,1].set_ylabel('Predicted Sentiment Score')
    axes[0,1].set_title(f'Best Model Predictions ({best_name})\nR² = {final_r2:.3f}')
    axes[0,1].grid(True, alpha=0.3)
    
    
    residuals = y_test - y_pred_final
    axes[1,0].scatter(y_pred_final, residuals, alpha=0.6, color='orange')
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Values')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residuals Plot')
    axes[1,0].grid(True, alpha=0.3)
    
    
    axes[1,1].hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Residuals Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/enhanced_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    if best_model is not None:
        cv_scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='r2')
        print(f"\nCross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    else:
        print("\nCross-validation skipped due to model training issues")
    
    return best_model, final_r2, final_rmse

def create_summary_report(monthly_scores, flight_risk_df, model_r2):
    """
    Create summary report for README
    """
    print("\n" + "=" * 60)
    print("CREATING SUMMARY REPORT")
    print("=" * 60)
    
    
    overall_scores = monthly_scores.groupby('employee')['monthly_sentiment_score'].mean().sort_values(ascending=False)
    
    top_3_positive = overall_scores.head(3)
    top_3_negative = overall_scores.tail(3)
    
    print("Overall Top 3 Positive Employees:")
    for i, (emp, score) in enumerate(top_3_positive.items(), 1):
        print(f"{i}. {emp} (Score: {score:.1f})")
    
    print("\nOverall Top 3 Negative Employees:")
    for i, (emp, score) in enumerate(top_3_negative.items(), 1):
        print(f"{i}. {emp} (Score: {score:.1f})")
    
    print(f"\nFlight Risk Employees ({len(flight_risk_df)} total):")
    for emp in flight_risk_df['employee']:
        print(f"- {emp}")
    
    print(f"\nModel Performance: R² = {model_r2:.3f}")
    
    readme_content = f"""# Employee Sentiment & Flight Risk Analysis



## Summary

This project analyzes employee communication data to identify sentiment trends and potential flight risks.

###  Top 3 Positive Employees:
1. {top_3_positive.index[0]} (Score: {top_3_positive.iloc[0]:.1f})
2. {top_3_positive.index[1]} (Score: {top_3_positive.iloc[1]:.1f})
3. {top_3_positive.index[2]} (Score: {top_3_positive.iloc[2]:.1f})

###  Top 3 Negative Employees:
1. {top_3_negative.index[0]} (Score: {top_3_negative.iloc[0]:.1f})
2. {top_3_negative.index[1]} (Score: {top_3_negative.iloc[1]:.1f})
3. {top_3_negative.index[2]} (Score: {top_3_negative.iloc[2]:.1f})

###  Employees Flagged as Flight Risks:
"""
    
    for emp in flight_risk_df['employee']:
        readme_content += f"- {emp}\n"
    
    readme_content += f"""
###  Key Insights:
- Model Performance: R² = {model_r2:.3f}
- {len(flight_risk_df)} employees identified as flight risks
- Sentiment analysis completed using VADER
- Monthly scoring system implemented
- Predictive modeling with linear regression

###  Files Generated:
- `monthly_employee_sentiment_scores.xlsx`: Monthly sentiment scores
- `employee_monthly_ranking.xlsx`: Monthly employee rankings
- `flight_risk_employees.xlsx`: Flight risk employee list
- `visualization/`: Charts and graphs
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("README.md updated with summary!")

def main():
    """
    Main execution function
    """
    print("EMPLOYEE SENTIMENT ANALYSIS PROJECT")
    print("=" * 60)
    
    
    import os
    os.makedirs('visualization', exist_ok=True)
    
    df = load_and_clean_data()
    df = perform_sentiment_labeling(df)
    df = exploratory_data_analysis(df)
    monthly_scores = calculate_employee_scores(df)
    employee_ranking(monthly_scores)
    flight_risk_df = flight_risk_identification(df)
    model, r2, rmse = predictive_modeling(df)
    create_summary_report(monthly_scores, flight_risk_df, r2)
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("All tasks completed according to project requirements:")
    print(" Task 1: Sentiment Labeling")
    print(" Task 2: Exploratory Data Analysis")
    print(" Task 3: Employee Score Calculation")
    print(" Task 4: Employee Ranking")
    print(" Task 5: Flight Risk Identification (CORRECTED)")
    print(" Task 6: Predictive Modeling")
    print(" Documentation and Visualizations")

if __name__ == "__main__":
    main() 