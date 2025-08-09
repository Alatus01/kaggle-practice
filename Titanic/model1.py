import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering
def process_data(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
    
    # Convert categorical to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df

train_processed = process_data(train)
test_processed = process_data(test)

# Features to use
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']

# Prepare data
X_train = train_processed[features]
y_train = train_processed['Survived']
X_test = test_processed[features]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Create submission file
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)