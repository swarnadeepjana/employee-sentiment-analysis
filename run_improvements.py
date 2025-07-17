# Run Model Improvements
from model_improvements import improve_sentiment_model

print("=" * 60)
print("EMPLOYEE SENTIMENT ANALYSIS - MODEL IMPROVEMENT")
print("=" * 60)

# Run the improvement function
best_model, scaler, feature_columns, final_r2 = improve_sentiment_model()

print("\n" + "=" * 60)
print("IMPROVEMENT SUMMARY")
print("=" * 60)
print(f"Original R² Score: 0.617")
print(f"Improved R² Score: {final_r2:.3f}")
print(f"Improvement: {((final_r2 - 0.617) / 0.617 * 100):.1f}%")

if final_r2 > 0.617:
    print("✅ SUCCESS: Model performance improved!")
else:
    print("⚠️  WARNING: Model performance did not improve")

print("\nKey improvements implemented:")
print("1. Enhanced feature engineering (19 features vs 3 original)")
print("2. Multiple model comparison (7 different algorithms)")
print("3. Hyperparameter tuning with GridSearchCV")
print("4. Feature scaling and outlier removal")
print("5. Cross-validation for robust evaluation")
print("6. Advanced text analysis features")
print("7. Time-based and rolling window features") 