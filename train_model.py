import pandas as pd
import numpy as np
import os
import glob
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load dataset from multiple possible formats and locations
    """
    print("🔍 Searching for dataset files...")
    
    # Get current directory
    folder = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Looking in folder: {folder}")
    
    # List all possible file patterns
    file_patterns = [
        "indian_salary_data_500.*",
        "salary_data.*",
        "indian_salary.*",
        "employee_salary.*",
        "data.*"
    ]
    
    # List all supported file extensions
    supported_extensions = ['.csv', '.xls', '.xlsx', '.json']
    
    # Search for files
    found_files = []
    for pattern in file_patterns:
        files = glob.glob(os.path.join(folder, pattern))
        found_files.extend(files)
    
    # Also check parent directory
    parent_folder = os.path.dirname(folder)
    for pattern in file_patterns:
        files = glob.glob(os.path.join(parent_folder, pattern))
        found_files.extend(files)
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    print(f"📄 Found {len(found_files)} potential files:")
    for file in found_files:
        print(f"   - {os.path.basename(file)}")
    
    # Try to load each file
    for file_path in found_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in supported_extensions:
            print(f"\n📂 Attempting to load: {os.path.basename(file_path)}")
            
            try:
                # Load based on file type
                if file_ext == '.csv':
                    df = pd.read_csv(file_path)
                    print("✅ CSV file loaded successfully.")
                    
                elif file_ext == '.xlsx':
                    df = pd.read_excel(file_path, engine='openpyxl')
                    print("✅ XLSX file loaded successfully.")
                    
                elif file_ext == '.xls':
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                        print("✅ XLS file loaded successfully.")
                    except:
                        # Fallback to openpyxl
                        df = pd.read_excel(file_path, engine='openpyxl')
                        print("✅ XLS file loaded successfully (using openpyxl).")
                        
                elif file_ext == '.json':
                    df = pd.read_json(file_path)
                    print("✅ JSON file loaded successfully.")
                
                # Validate the dataframe
                if validate_dataframe(df):
                    return df
                else:
                    print("❌ File doesn't contain required columns.")
                    continue
                    
            except Exception as e:
                print(f"❌ Error loading {os.path.basename(file_path)}: {str(e)}")
                continue
    
    # If no file found, create sample data
    print("\n❌ No valid dataset file found!")
    print("🔧 Creating sample dataset for demonstration...")
    return create_sample_data()

def validate_dataframe(df):
    """
    Validate if dataframe has required columns
    """
    required_columns = [
        'age', 'gender', 'education', 'years_of_experience',
        'job_title', 'job_location', 'city', 'nationality', 'salary_inr_lakhs'
    ]
    
    # Check if all required columns exist (case insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    required_columns_lower = [col.lower() for col in required_columns]
    
    missing_columns = []
    for req_col in required_columns_lower:
        if req_col not in df_columns_lower:
            missing_columns.append(req_col)
    
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        print(f"📋 Available columns: {list(df.columns)}")
        return False
    
    # Rename columns to standard format (if needed)
    column_mapping = {}
    for i, col in enumerate(df.columns):
        if col.lower() in required_columns_lower:
            correct_name = required_columns[required_columns_lower.index(col.lower())]
            if col != correct_name:
                column_mapping[col] = correct_name
    
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
        print(f"📝 Renamed columns: {column_mapping}")
    
    print(f"✅ Dataframe validated. Shape: {df.shape}")
    return True

def create_sample_data():
    """
    Create sample data for demonstration purposes
    """
    print("🎲 Generating sample Indian salary data...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Sample data generation
    ages = np.random.randint(22, 60, n_samples)
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3])
    
    education_levels = [
        'Bachelor\'s Degree', 'Master\'s Degree', 'PhD', 'Diploma',
        'High School', 'Associate Degree'
    ]
    education = np.random.choice(education_levels, n_samples, 
                                p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    
    experience = np.random.randint(0, 30, n_samples)
    
    job_titles = [
        'Software Engineer', 'Data Scientist', 'Product Manager', 
        'Business Analyst', 'DevOps Engineer', 'UI/UX Designer',
        'Marketing Manager', 'Sales Manager', 'HR Manager',
        'Financial Analyst', 'Consultant', 'Project Manager'
    ]
    job_title = np.random.choice(job_titles, n_samples)
    
    job_locations = ['On-site', 'Remote', 'Hybrid']
    job_location = np.random.choice(job_locations, n_samples, p=[0.5, 0.3, 0.2])
    
    cities = [
        'Bangalore', 'Mumbai', 'Delhi', 'Pune', 'Hyderabad',
        'Chennai', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Gurgaon'
    ]
    city = np.random.choice(cities, n_samples)
    
    nationalities = ['Indian', 'American', 'British', 'Canadian', 'Australian']
    nationality = np.random.choice(nationalities, n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05])
    
    # Calculate salary based on factors
    base_salary = 5 + (experience * 0.8) + (ages - 22) * 0.2
    
    # Add job title multipliers
    job_multipliers = {
        'Software Engineer': 1.2, 'Data Scientist': 1.5, 'Product Manager': 1.4,
        'Business Analyst': 1.0, 'DevOps Engineer': 1.3, 'UI/UX Designer': 1.1,
        'Marketing Manager': 1.2, 'Sales Manager': 1.1, 'HR Manager': 1.0,
        'Financial Analyst': 1.1, 'Consultant': 1.3, 'Project Manager': 1.2
    }
    
    # Add city multipliers
    city_multipliers = {
        'Bangalore': 1.3, 'Mumbai': 1.4, 'Delhi': 1.3, 'Pune': 1.2,
        'Hyderabad': 1.2, 'Chennai': 1.1, 'Kolkata': 1.0, 'Ahmedabad': 1.0,
        'Jaipur': 0.9, 'Gurgaon': 1.3
    }
    
    # Add education multipliers
    edu_multipliers = {
        'PhD': 1.5, 'Master\'s Degree': 1.3, 'Bachelor\'s Degree': 1.0,
        'Associate Degree': 0.8, 'Diploma': 0.7, 'High School': 0.6
    }
    
    # Calculate final salary
    salary = []
    for i in range(n_samples):
        sal = base_salary[i]
        sal *= job_multipliers.get(job_title[i], 1.0)
        sal *= city_multipliers.get(city[i], 1.0)
        sal *= edu_multipliers.get(education[i], 1.0)
        
        # Add some randomness
        sal *= np.random.uniform(0.8, 1.2)
        
        # Ensure minimum salary
        sal = max(sal, 2.5)
        salary.append(round(sal, 1))
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'education': education,
        'years_of_experience': experience,
        'job_title': job_title,
        'job_location': job_location,
        'city': city,
        'nationality': nationality,
        'salary_inr_lakhs': salary
    })
    
    # Save sample data
    df.to_csv('indian_salary_data_500.csv', index=False)
    print(f"💾 Sample data saved as 'indian_salary_data_500.csv'")
    print(f"📊 Generated {len(df)} records")
    
    return df

def preprocess_data(df):
    """
    Clean and preprocess the data
    """
    print("\n🧹 Preprocessing data...")
    
    print(f"📊 Initial data shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("⚠️ Missing values found:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"   - {col}: {missing} missing values")
        
        print("🔧 Handling missing values...")
        # Fill missing values or drop rows
        df = df.dropna()
        print(f"✅ After removing missing values: {df.shape}")
    
    # Remove outliers in salary
    initial_count = len(df)
    df = df[(df['salary_inr_lakhs'] >= 1) & (df['salary_inr_lakhs'] <= 150)]
    removed_outliers = initial_count - len(df)
    
    if removed_outliers > 0:
        print(f"🔧 Removed {removed_outliers} salary outliers")
        print(f"📊 Final data shape: {df.shape}")
    
    # Prepare features and target
    feature_columns = ['age', 'gender', 'education', 'years_of_experience',
                      'job_title', 'job_location', 'city', 'nationality']
    
    X = df[feature_columns].copy()
    y = df['salary_inr_lakhs'].copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'education', 'job_title', 'job_location', 'city', 'nationality']
    
    print("🔤 Encoding categorical variables...")
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"   - {col}: {len(le.classes_)} unique values")
    
    print(f"✅ Preprocessing complete. Features shape: {X.shape}")
    return X, y, label_encoders

def train_model(X, y):
    """
    Train the machine learning model
    """
    print("\n📈 Training machine learning model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Initialize and train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )
    
    print("🤖 Training Gradient Boosting Regressor...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\n📊 Model Performance Metrics:")
    print(f"   🎯 Training R² Score: {train_r2:.4f} ({train_r2*100:.1f}%)")
    print(f"   🎯 Test R² Score: {test_r2:.4f} ({test_r2*100:.1f}%)")
    print(f"   📉 Mean Absolute Error: ₹{test_mae:.2f} Lakhs")
    print(f"   📉 Root Mean Square Error: ₹{test_rmse:.2f} Lakhs")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top Feature Importances:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")
    
    return model, test_r2, feature_importance

def save_model(model, encoders, accuracy, feature_importance):
    """
    Save the trained model and related files
    """
    print("\n💾 Saving model and encoders...")
    
    try:
        # Save model
        joblib.dump(model, 'model.joblib')
        print("✅ Model saved as 'model.joblib'")
        
        # Save encoders
        joblib.dump(encoders, 'label_encoders.joblib')
        print("✅ Label encoders saved as 'label_encoders.joblib'")
        
        # Save model info
        model_info = {
            "accuracy": accuracy,
            "model_type": "GradientBoostingRegressor",
            "features": list(encoders.keys()) + ['age', 'years_of_experience'],
            "date_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        joblib.dump(model_info, 'model_info.joblib')
        print("✅ Model info saved as 'model_info.joblib'")
        
        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("✅ Feature importance saved as 'feature_importance.csv'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return False

def main():
    """
    Main function to orchestrate the training process
    """
    print("🚀 Starting Pay Predictor Model Training")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        df = load_and_prepare_data()
        if df is None:
            print("❌ Failed to load data. Exiting...")
            sys.exit(1)
        
        # Step 2: Preprocess data
        X, y, encoders = preprocess_data(df)
        
        # Step 3: Train model
        model, accuracy, feature_importance = train_model(X, y)
        
        # Step 4: Save model
        save_success = save_model(model, encoders, accuracy, feature_importance)
        
        if save_success:
            print("\n" + "=" * 60)
            print(f"✅ Training Complete!")
            print(f"🎯 Model Accuracy: {accuracy*100:.1f}%")
            print(f"📁 Files created:")
            print(f"   - model.joblib")
            print(f"   - label_encoders.joblib") 
            print(f"   - model_info.joblib")
            print(f"   - feature_importance.csv")
            if not os.path.exists('indian_salary_data_500.csv'):
                print(f"   - indian_salary_data_500.csv (sample data)")
            print("\n🎉 You can now run the Streamlit app!")
            print("💡 Command: streamlit run app.py")
        else:
            print("❌ Failed to save model files.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("🔧 Please check your data file and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()