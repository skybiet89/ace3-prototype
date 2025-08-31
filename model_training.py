
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_customer_data(csv_path='customer_data.csv'):
    """Load customer data from CSV file"""
    try:
        # Try to load from current directory first
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded data from {csv_path}")
            print(f"📊 Data shape: {df.shape}")
            return df
        else:
            # Try alternative paths that might work in Streamlit Cloud
            alternative_paths = [
                './customer_data.csv',
                'data/customer_data.csv',
                '../customer_data.csv'
            ]

            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    df = pd.read_csv(alt_path)
                    print(f"✅ Loaded data from {alt_path}")
                    print(f"📊 Data shape: {df.shape}")
                    return df

            # If no CSV found, create sample data as fallback
            print("⚠️ CSV file not found. Creating minimal sample data...")
            return create_fallback_data()

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        print("🔄 Creating fallback data...")
        return create_fallback_data()

def create_fallback_data():
    """Create minimal fallback data if CSV is not available"""
    import random
    random.seed(42)
    np.random.seed(42)

    data = []
    city_types = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
    recent_actions = ['cart abandonment', 'not visiting the app', 'not adding to cart', 'wrote a bad review']
    categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Health', 'Sports', 'Beauty']
    devices = ['Mobile', 'Desktop', 'Tablet']
    locales = ['en_US', 'en_GB', 'es_ES', 'fr_FR', 'de_DE']

    # Create smaller dataset for fallback
    for i in range(500):
        customer = {
            'Customer_ID': f"CUST_{100000 + i}",
            'Age_of_Buyer': random.randint(16, 58),
            'Type_of_City': random.choice(city_types),
            'Average_Order_Value': random.randint(200, 10000),
            'Recent_Action': random.choice(recent_actions),
            'Time_Spent_on_App_per_Day_minutes': random.randint(5, 180),
            'Email': f"customer{i}@example.com",
            'Most_Active_Time_of_User': f"{random.randint(1,12)}:00 {random.choice(['AM','PM'])}",
            'Recency_days': random.randint(1, 365),
            'Frequency_purchases': random.randint(1, 50),
            'Monetary_spend': random.randint(100, 20000),
            'Email_Open_Rate': round(random.random(), 3),
            'Push_CTR': round(random.random(), 3),
            'Coupon_Redemption_Rate': round(random.random(), 3),
            'Chat_Engagement_Rate': round(random.random(), 3),
            'Top_Category': random.choice(categories),
            'Primary_Device': random.choice(devices),
            'Locale': random.choice(locales),
            'CLTV': round(random.uniform(50, 5000), 2),
            'Pages_per_Session': round(random.uniform(1, 20), 1),
            'Avg_Session_Duration_min': round(random.uniform(1, 20), 1),
            'Product_Views': random.randint(1, 10),
            'Discount_Threshold_percent': random.choice([5, 10, 15, 20, 25, 30]),
            'Churn_Risk': round(random.random(), 3)
        }
        data.append(customer)

    df = pd.DataFrame(data)

    # Add marketing suggestions using the same logic
    def get_marketing_suggestion(row):
        recent_action = row['Recent_Action']
        email_open_rate = row['Email_Open_Rate']
        push_ctr = row['Push_CTR']
        chat_engagement = row['Chat_Engagement_Rate']
        coupon_redemption = row['Coupon_Redemption_Rate']
        churn_risk = row['Churn_Risk']
        cltv = row['CLTV']

        if recent_action == 'cart abandonment':
            if coupon_redemption > 0.7:
                return 'Discount Coupon Email'
            elif push_ctr > 0.5:
                return 'Push Notification Reminder'
            else:
                return 'Abandoned Cart Email'
        elif recent_action == 'not visiting the app':
            if churn_risk > 0.8:
                return 'Win-back Campaign'
            else:
                return 'Re-engagement Push'
        elif recent_action == 'wrote a bad review':
            if chat_engagement > 0.7:
                return 'Proactive Chat Support'
            else:
                return 'Manager Follow-up Call'
        else:  # not adding to cart
            if email_open_rate > 0.7:
                return 'Personalized Email Recommendations'
            else:
                return 'Browsing Behavior Email'

    df['Marketing_Suggestion'] = df.apply(get_marketing_suggestion, axis=1)
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    print("🔧 Preparing features for ML training...")

    # Define numerical features (using exact column names from CSV)
    feature_columns = [
        'Age_of_Buyer', 'Average_Order_Value', 'Time_Spent_on_App_per_Day_minutes', 
        'Recency_days', 'Frequency_purchases', 'Monetary_spend',
        'Email_Open_Rate', 'Push_CTR', 'Coupon_Redemption_Rate', 'Chat_Engagement_Rate',
        'CLTV', 'Pages_per_Session', 'Avg_Session_Duration_min', 'Product_Views',
        'Discount_Threshold_percent', 'Churn_Risk'
    ]

    # Create label encoders for categorical variables
    encoders = {}

    # Encode categorical features
    categorical_features = {
        'Type_of_City': 'City_Encoded',
        'Recent_Action': 'Action_Encoded', 
        'Top_Category': 'Category_Encoded',
        'Primary_Device': 'Device_Encoded',
        'Locale': 'Locale_Encoded'
    }

    for original_col, encoded_col in categorical_features.items():
        if original_col in df.columns:
            le = LabelEncoder()
            df[encoded_col] = le.fit_transform(df[original_col])
            encoders[original_col.lower() + '_encoder'] = le
            feature_columns.append(encoded_col)
        else:
            print(f"⚠️ Warning: {original_col} not found in dataset")

    return df, feature_columns, encoders

def train_model(df, feature_columns, encoders):
    """Train the Random Forest model"""
    print("🤖 Training Random Forest model...")

    # Prepare features and target
    X = df[feature_columns]
    y = df['Marketing_Suggestion']

    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🎯 Unique marketing actions: {y.nunique()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained successfully!")
    print(f"🎯 Training accuracy: {accuracy:.3f}")
    print(f"📊 Number of features: {len(feature_columns)}")
    print(f"🏷️ Number of classes: {len(model.classes_)}")

    return model, accuracy

def save_model_components(model, encoders, feature_columns):
    """Save model, encoders, and feature columns"""
    print("💾 Saving model components...")

    try:
        # Save model
        joblib.dump(model, 'model.pkl')
        print("✅ Model saved to model.pkl")

        # Save encoders
        joblib.dump(encoders, 'encoders.pkl') 
        print("✅ Encoders saved to encoders.pkl")

        # Save feature columns
        joblib.dump(feature_columns, 'feature_columns.pkl')
        print("✅ Feature columns saved to feature_columns.pkl")

        return True

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def prepare_and_train_model():
    """Main function to prepare data and train model"""
    print("🚀 Starting ML model training pipeline...")
    print("=" * 50)

    try:
        # Load data from CSV
        df = load_customer_data()

        if df is None or df.empty:
            print("❌ No data available for training")
            return None, None, None

        # Data validation
        required_columns = ['Marketing_Suggestion', 'Age_of_Buyer', 'Type_of_City', 'Recent_Action']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return None, None, None

        # Prepare features
        df, feature_columns, encoders = prepare_features(df)

        # Train model
        model, accuracy = train_model(df, feature_columns, encoders)

        # Save components
        save_success = save_model_components(model, encoders, feature_columns)

        if save_success:
            print("=" * 50)
            print("🎉 Model training completed successfully!")
            print(f"📈 Final accuracy: {accuracy:.1%}")
            print(f"🎯 Marketing actions: {len(model.classes_)}")
            print("🚀 Ready for predictions!")

            return model, encoders, feature_columns
        else:
            print("❌ Failed to save model components")
            return None, None, None

    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    prepare_and_train_model()
