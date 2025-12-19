import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_models(model_dir='outputs'):
    with open(os.path.join(model_dir, 'risk_model.pkl'), 'rb') as f:
        risk_model = pickle.load(f)
    with open(os.path.join(model_dir, 'crime_model.pkl'), 'rb') as f:
        crime_model = pickle.load(f)
    with open(os.path.join(model_dir, 'risk_encoder.pkl'), 'rb') as f:
        risk_encoder = pickle.load(f)
    with open(os.path.join(model_dir, 'crime_encoder.pkl'), 'rb') as f:
        crime_encoder = pickle.load(f)
    return risk_model, crime_model, risk_encoder, crime_encoder

def load_location_encoder():
    features_path = 'data/features.csv'
    if os.path.exists(features_path):
        df = pd.read_csv(features_path, low_memory=False)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(df['location'].unique())
        return le
    return None

def get_crime_statistics(location, features_path='data/features.csv'):
    if not os.path.exists(features_path):
        return {}
    
    df = pd.read_csv(features_path, low_memory=False)
    location_data = df[df['location'] == location]
    
    if len(location_data) == 0:
        return {}
    
    crime_counts = location_data['crime_type'].value_counts().to_dict()
    total_crimes = len(location_data)
    
    crime_freq = {}
    for crime_type, count in crime_counts.items():
        if pd.isna(crime_type) or crime_type == 'nan' or str(crime_type) == 'nan':
            continue
        freq_pct = (count / total_crimes) * 100
        if freq_pct > 30:
            freq_label = 'Very Common'
        elif freq_pct > 15:
            freq_label = 'Common'
        elif freq_pct > 5:
            freq_label = 'Occasional'
        else:
            freq_label = 'Rare'
        
        crime_freq[crime_type] = {
            'count': int(count),
            'percentage': round(freq_pct, 2),
            'frequency': freq_label
        }
    
    most_common = None
    if len(crime_counts) > 0:
        most_common = list(crime_counts.keys())[0]
    
    return {
        'total_crimes': total_crimes,
        'crime_frequency': crime_freq,
        'most_common': most_common
    }

def create_features_from_input(location, day_of_week, hour, month, year):
    location_encoder = load_location_encoder()
    if location_encoder is None:
        raise ValueError("Location encoder not found. Run feature engineering first.")
    
    try:
        location_encoded = location_encoder.transform([location])[0]
    except:
        location_encoded = 0
    
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'year': [year],
        'location_encoded': [location_encoded],
        'crime_type_encoded': [0],
        'crime_type_frequency': [0],
        'past_crime_count': [0],
        'crime_count_30d': [0]
    })
    
    return features

def predict_comprehensive(location, day_of_week, hour, month=None, year=None, model_dir='outputs'):
    if month is None or year is None:
        now = datetime.now()
        month = month if month is not None else now.month
        year = year if year is not None else now.year
    
    risk_model, crime_model, risk_encoder, crime_encoder = load_models(model_dir)
    
    features = create_features_from_input(location, day_of_week, hour, month, year)
    
    risk_pred = risk_model.predict(features)[0]
    risk_proba = risk_model.predict_proba(features)[0]
    risk_level = risk_encoder.inverse_transform([risk_pred])[0]
    risk_probabilities = dict(zip(risk_encoder.classes_, risk_proba))
    
    crime_pred = crime_model.predict(features)[0]
    crime_proba = crime_model.predict_proba(features)[0]
    crime_type = crime_encoder.inverse_transform([crime_pred])[0]
    crime_probabilities = dict(zip(crime_encoder.classes_, crime_proba))
    
    stats = get_crime_statistics(location)
    
    result = {
        'risk_level': risk_level,
        'risk_probabilities': risk_probabilities,
        'predicted_crime_type': crime_type,
        'crime_type_probability': float(crime_proba[crime_pred]),
        'crime_type_probabilities': {k: float(v) for k, v in crime_probabilities.items()},
        'location': location,
        'crime_statistics': stats
    }
    
    return result

def format_percentage(value):
    return f"{value * 100:.1f}%"

def format_number(num):
    return f"{num:,}"

def create_bar_chart(value, max_value=1.0, width=20):
    filled = int((value / max_value) * width)
    return "█" * filled

def print_section_header(title, width=80):
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_subsection_header(title, width=80):
    print("\n" + title)
    print("─" * width)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 80)
        print("CRIME RISK PREDICTION SYSTEM - USAGE".center(80))
        print("=" * 80)
        print("\nUsage: python 05_predict.py <location> <day_of_week> <hour> [month] [year]")
        print("\nArguments:")
        print("  location      : City and State in quotes (e.g., 'Chicago, IL')")
        print("  day_of_week   : Day of week (0-6)")
        print("                 0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday")
        print("                 4 = Friday, 5 = Saturday, 6 = Sunday")
        print("  hour          : Hour of day in 24-hour format (0-23)")
        print("                 0 = Midnight, 12 = Noon, 23 = 11:00 PM")
        print("  month         : Month (1-12), optional, defaults to current month")
        print("                 1 = January, 12 = December")
        print("  year          : Year (4 digits), optional, defaults to current year")
        print("\nExamples:")
        print("  python 05_predict.py 'Chicago, IL' 0 14")
        print("  python 05_predict.py 'Chicago, IL' 0 14 3 2024")
        print("  python 05_predict.py 'Los Angeles, CA' 5 20 12 2023")
        print("\n" + "=" * 80)
        sys.exit(1)
    
    location = sys.argv[1]
    day_of_week = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hour = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    month = int(sys.argv[4]) if len(sys.argv) > 4 else None
    year = int(sys.argv[5]) if len(sys.argv) > 5 else None
    
    if day_of_week < 0 or day_of_week > 6:
        print(f"Error: day_of_week must be between 0-6, got {day_of_week}")
        print("  0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday")
        print("  4 = Friday, 5 = Saturday, 6 = Sunday")
        sys.exit(1)
    
    if hour < 0 or hour > 23:
        print(f"Error: hour must be between 0-23, got {hour}")
        sys.exit(1)
    
    if month is not None and (month < 1 or month > 12):
        print(f"Error: month must be between 1-12, got {month}")
        sys.exit(1)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    result = predict_comprehensive(location, day_of_week, hour, month, year)
    
    print_section_header("CRIME RISK PREDICTION SYSTEM")
    
    print_subsection_header("INPUT PARAMETERS")
    print(f"Location:        {location}")
    print(f"Day of Week:     {day_names[day_of_week]} ({day_of_week})")
    hour_12 = hour % 12
    hour_12 = 12 if hour_12 == 0 else hour_12
    am_pm = "AM" if hour < 12 else "PM"
    print(f"Hour:            {hour:02d}:00 ({hour_12}:00 {am_pm})")
    if month:
        print(f"Month:           {month_names[month-1]} ({month})")
    if year:
        print(f"Year:            {year}")
    
    print_section_header("PREDICTION RESULTS")
    
    risk_level = result['risk_level']
    risk_symbol = "⚠️" if risk_level == "High" else "⚡" if risk_level == "Medium" else "✓"
    print(f"\nRISK LEVEL: [{risk_level.upper()}] {risk_symbol}")
    print_subsection_header("")
    
    risk_probs = result['risk_probabilities']
    predicted_risk = result['risk_level']
    for risk in ['Low', 'Medium', 'High']:
        prob = risk_probs.get(risk, 0)
        marker = "  ← PREDICTED" if risk == predicted_risk else ""
        print(f"{risk:12s} Risk:  {format_percentage(prob):>6s}{marker}")
    
    print_subsection_header("")
    crime_type = result['predicted_crime_type']
    crime_conf = result['crime_type_probability']
    print(f"PREDICTED CRIME TYPE: {crime_type}")
    print(f"Confidence: {format_percentage(crime_conf)}")
    print_subsection_header("")
    
    crime_probs = result['crime_type_probabilities']
    sorted_crimes = sorted(crime_probs.items(), key=lambda x: x[1], reverse=True)
    top_crimes = sorted_crimes[:10]
    max_prob = max(crime_probs.values()) if crime_probs else 1.0
    
    print("Top Crime Type Probabilities:")
    for crime, prob in top_crimes:
        bar = create_bar_chart(prob, max_prob, 20)
        print(f"  {crime:20s}  {format_percentage(prob):>6s}  {bar}")
    
    print_section_header("HISTORICAL STATISTICS")
    
    stats = result['crime_statistics']
    if stats:
        total = stats.get('total_crimes', 0)
        most_common = stats.get('most_common', 'N/A')
        print(f"Total Crimes in History: {format_number(total)}")
        print(f"Most Common Crime Type: {most_common}")
        
        crime_freq = stats.get('crime_frequency', {})
        if crime_freq:
            print("\nTop Crime Types by Frequency:")
            sorted_freq = sorted(crime_freq.items(), key=lambda x: x[1]['count'], reverse=True)
            for crime, info in sorted_freq[:10]:
                count = info['count']
                pct = info['percentage']
                freq_label = info['frequency']
                print(f"  {crime:20s}  {format_number(count):>8s} ({pct:5.1f}%)  [{freq_label}]")
    else:
        print("No historical statistics available for this location.")
    
    print("\n" + "=" * 80)
