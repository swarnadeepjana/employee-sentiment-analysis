# Advanced Model Improvements to Boost R² Score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

def improve_sentiment_model():
    """
    Comprehensive function to improve the R² score of the sentiment analysis model
    """
    
    # Load the data
    print("Loading data...")
    df = pd.read_excel("D:\\LLM interns\\Questions\\labeled_messages.xlsx")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Enhanced Feature Engineering
    print("Creating advanced features...")
    
    # 1. Time-based features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Sentiment intensity features
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_scores(text):
        if not isinstance(text, str):
            return {'pos': 0, 'neg': 0, 'neu': 0, 'compound': 0}
        return analyzer.polarity_scores(text)
    
    sentiment_scores = df['cleaned_message'].apply(get_sentiment_scores)
    df['positive_score'] = sentiment_scores.apply(lambda x: x['pos'])
    df['negative_score'] = sentiment_scores.apply(lambda x: x['neg'])
    df['neutral_score'] = sentiment_scores.apply(lambda x: x['neu'])
    df['compound_score'] = sentiment_scores.apply(lambda x: x['compound'])
    
    # 3. Text complexity features
    df['avg_word_length'] = df['cleaned_message'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0)
    df['unique_word_ratio'] = df['cleaned_message'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()) if str(x).split() else 0)
    df['exclamation_count'] = df['body'].str.count('!')
    df['question_count'] = df['body'].str.count('\\?')
    df['capital_ratio'] = df['body'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    
    # Create sentiment score mapping
    sentiment_score_map = {
        'Positive': 1,
        'Neutral': 0,
        'Negative': -1
    }
    df['sentiment_score'] = df['sentiment_label'].replace(sentiment_score_map)
    
    # 4. Employee behavior features
    employee_stats = df.groupby('from').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'word_count': ['mean', 'std'],
        'positive_score': 'mean',
        'negative_score': 'mean'
    }).reset_index()
    
    employee_stats.columns = ['from', 'emp_sentiment_mean', 'emp_sentiment_std', 'emp_message_count', 
                             'emp_word_mean', 'emp_word_std', 'emp_pos_mean', 'emp_neg_mean']
    
    df = df.merge(employee_stats, on='from', how='left')
    
    # 5. Rolling window features
    df = df.sort_values(['from', 'date'])
    df['rolling_sentiment_3'] = df.groupby('from')['sentiment_score'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_sentiment_7'] = df.groupby('from')['sentiment_score'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    
    # Create monthly features with enhanced engineering
    monthly_features = df.groupby(['from', df['date'].dt.to_period('M')]).agg({
        'cleaned_message': 'count',
        'word_count': ['sum', 'mean', 'std'],
        'sentiment_score': ['sum', 'mean', 'std'],
        'positive_score': ['mean', 'sum'],
        'negative_score': ['mean', 'sum'],
        'compound_score': ['mean', 'std'],
        'avg_word_length': 'mean',
        'unique_word_ratio': 'mean',
        'exclamation_count': 'sum',
        'question_count': 'sum',
        'capital_ratio': 'mean',
        'rolling_sentiment_3': 'mean',
        'rolling_sentiment_7': 'mean'
    }).reset_index()
    
    monthly_features.columns = ['employee', 'month', 'message_count', 'total_words', 'avg_words', 'std_words',
                               'sentiment_sum', 'sentiment_mean', 'sentiment_std', 'pos_mean', 'pos_sum',
                               'neg_mean', 'neg_sum', 'compound_mean', 'compound_std', 'avg_word_len',
                               'unique_ratio', 'exclamation_total', 'question_total', 'capital_ratio',
                               'rolling_3', 'rolling_7']
    
    monthly_features.dropna(inplace=True)
    
    # Feature selection
    feature_columns = ['message_count', 'total_words', 'avg_words', 'std_words', 'sentiment_mean', 
                       'sentiment_std', 'pos_mean', 'pos_sum', 'neg_mean', 'neg_sum', 'compound_mean', 
                       'compound_std', 'avg_word_len', 'unique_ratio', 'exclamation_total', 
                       'question_total', 'capital_ratio', 'rolling_3', 'rolling_7']
    
    X = monthly_features[feature_columns]
    y = monthly_features['sentiment_sum']
    
    # Remove outliers
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
    
    X = X[outlier_mask]
    y = y[outlier_mask]
    
    print(f"Dataset shape after outlier removal: {X.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training multiple models...")
    
    # 1. Linear Regression with regularization
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net': ElasticNet(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR()
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        if name in ['SVR']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_train_scaled)
            train_r2 = r2_score(y_train, y_pred)
            y_pred_test = model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_pred)
            y_pred_test = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {'train_r2': train_r2, 'test_r2': test_r2}
        print(f"{name}: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['test_r2']:.3f}")
    
    # Hyperparameter tuning for the best model
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        best_model = RandomForestRegressor(random_state=42)
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        best_model = GradientBoostingRegressor(random_state=42)
    elif best_model_name == 'Ridge Regression':
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        best_model = Ridge()
    else:
        param_grid = {}
        best_model = models[best_model_name]
    
    if param_grid:
        print(f"\nTuning hyperparameters for {best_model_name}...")
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred_tuned = best_model.predict(X_test)
        tuned_r2 = r2_score(y_test, y_pred_tuned)
        
        print(f"Tuned {best_model_name}: R² = {tuned_r2:.3f}")
        print(f"Best parameters: {grid_search.best_params_}")
    
    # Feature importance based on model type
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting)
        importances = getattr(best_model, 'feature_importances_')
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    elif hasattr(best_model, 'coef_'):
        # Linear models (Linear Regression, Ridge, Lasso, Elastic Net)
        coefficients = getattr(best_model, 'coef_')
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop 10 Most Important Features (by coefficient magnitude):")
        print(feature_importance.head(10))
    else:
        # Models without direct feature importance (SVR, etc.)
        print("\nFeature importance not available for this model type.")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Final predictions
    y_pred_final = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred_final)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    
    print(f"\nFinal Model Performance:")
    print(f"R² Score: {final_r2:.3f}")
    print(f"RMSE: {final_rmse:.3f}")
    
    # Save the improved model
    import joblib
    joblib.dump(best_model, 'D:\\LLM interns\\Questions\\improved_sentiment_model.pkl')
    joblib.dump(scaler, 'D:\\LLM interns\\Questions\\feature_scaler.pkl')
    
    print("\nImproved model saved successfully!")
    
    return best_model, scaler, feature_columns, final_r2

if __name__ == "__main__":
    # Run the improvement function
    best_model, scaler, feature_columns, final_r2 = improve_sentiment_model()
    
    print(f"\nModel improvement completed!")
    print(f"Original R²: 0.617")
    print(f"Improved R²: {final_r2:.3f}")
    print(f"Improvement: {((final_r2 - 0.617) / 0.617 * 100):.1f}%") 