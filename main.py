import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score
from xgboost import XGBClassifier
import joblib

# Load data
file_path = 'data/premier-league-matches.csv'
matches = pd.read_csv(file_path)

# Clean data types
matches["date"] = pd.to_datetime(matches["Date"])
matches["h/a"] = matches["Home"].astype("category").cat.codes
matches["opp"] = matches["Away"].astype("category").cat.codes
matches["day"] = matches["date"].dt.dayofweek
matches["target"] = (matches["FTR"] == "H").astype("int")

# Adding recent team performance
matches['home_form'] = matches.groupby('Home')['target'].transform(lambda x: x.rolling(5, min_periods=1).mean())
matches['away_form'] = matches.groupby('Away')['target'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Ensure no NaN values in home_form and away_form
matches['home_form'].fillna(0.5, inplace=True)
matches['away_form'].fillna(0.5, inplace=True)

# Train-test split
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] >= '2022-01-01']
predictors = ["h/a", "opp", "day", "home_form", "away_form"]

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=1, class_weight='balanced')
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(train[predictors], train['target'])
best_rf = grid_search.best_estimator_

# Training XGBoost model
xgb = XGBClassifier()
xgb.fit(train[predictors], train['target'])

# Stacking Classifier
estimators = [
    ('rf', best_rf),
    ('xgb', xgb)
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(train[predictors], train['target'])

# Cross-validation
cv_scores = cross_val_score(stacking_clf, train[predictors], train['target'], cv=10)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

# Save the model
model_path = 'models/stacking_model.pkl'
joblib.dump(stacking_clf, model_path)

# Save encoders
home_encoder = LabelEncoder().fit(matches['Home'])
away_encoder = LabelEncoder().fit(matches['Away'])
joblib.dump(home_encoder, 'models/home_encoder.pkl')
joblib.dump(away_encoder, 'models/away_encoder.pkl')

# Make predictions
preds = stacking_clf.predict(test[predictors])

# Evaluate the model
accuracy = accuracy_score(test["target"], preds)
report = classification_report(test["target"], preds, target_names=['Loss/Draw', 'Win'])
precision = precision_score(test["target"], preds)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print(f"Precision: {precision}")

# Function to predict the outcome of a single match
def predict_match(home_team, away_team):
    home_encoder = joblib.load('models/home_encoder.pkl')
    away_encoder = joblib.load('models/away_encoder.pkl')
    home_encoded = home_encoder.transform([home_team])[0]
    away_encoded = away_encoder.transform([away_team])[0]
    
    # Ensure 'home_form' and 'away_form' are available for prediction
    if home_team in matches['Home'].values:
        recent_home_form = matches[matches['Home'] == home_team]['home_form'].mean()
    else:
        recent_home_form = 0.5  # or some default value
    
    if away_team in matches['Away'].values:
        recent_away_form = matches[matches['Away'] == away_team]['away_form'].mean()
    else:
        recent_away_form = 0.5  # or some default value
    
    # Create a DataFrame to ensure correct feature names and order
    input_data = pd.DataFrame([[home_encoded, away_encoded, 0, recent_home_form, recent_away_form]], 
                              columns=predictors)
    
    stacking_clf = joblib.load(model_path)
    prediction = stacking_clf.predict(input_data)[0]
    result_mapping = {1: 'Home Win', 0: 'Loss/Draw'}
    return result_mapping[prediction]

# Example usage
home_team = 'Burnley'
away_team = 'Chelsea'
prediction = predict_match(home_team, away_team)

# Detailed Output
print(f"Match Prediction:")
print(f"  Home Team: {home_team}")
print(f"  Away Team: {away_team}")
print(f"  Predicted Outcome for {home_team}: {prediction}")