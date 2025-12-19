import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict_comprehensive, get_crime_statistics

def validate_prediction_consistency(location, test_cases):
    """Test multiple scenarios for the same location to check consistency."""
    print("=" * 80)
    print(f"VALIDATION: Testing {location}")
    print("=" * 80)
    
    results = []
    for case in test_cases:
        day, hour, month, year = case
        result = predict_comprehensive(location, day, hour, month, year)
        results.append({
            'day': day,
            'hour': hour,
            'month': month,
            'year': year,
            'risk': result['risk_level'],
            'risk_probs': result['risk_probabilities'],
            'crime_type': result['predicted_crime_type'],
            'crime_conf': result['crime_type_probability']
        })
    
    print("\nPrediction Results:")
    print("-" * 80)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for r in results:
        print(f"Day {r['day']} ({day_names[r['day']]}), Hour {r['hour']:02d}:00, "
              f"Month {r['month']}/{r['year']} → Risk: {r['risk']}, "
              f"Crime: {r['crime_type']} ({r['crime_conf']:.1%})")
    
    # Check consistency
    print("\nConsistency Checks:")
    print("-" * 80)
    
    # 1. Check if probabilities sum to ~100%
    for r in results:
        prob_sum = sum(r['risk_probs'].values())
        if abs(prob_sum - 1.0) > 0.01:
            print(f"⚠️  Warning: Probabilities don't sum to 100% (sum={prob_sum:.2%})")
        else:
            print(f"✓ Probabilities sum correctly: {prob_sum:.2%}")
    
    # 2. Check if predicted risk has highest probability
    for r in results:
        predicted_risk = r['risk']
        predicted_prob = r['risk_probs'].get(predicted_risk, 0)
        max_prob = max(r['risk_probs'].values())
        if predicted_prob == max_prob:
            print(f"✓ Predicted risk '{predicted_risk}' has highest probability ({predicted_prob:.1%})")
        else:
            print(f"⚠️  Warning: Predicted risk '{predicted_risk}' ({predicted_prob:.1%}) "
                  f"is not the highest ({max_prob:.1%})")
    
    # 3. Compare night vs day
    night_results = [r for r in results if 20 <= r['hour'] <= 23 or 0 <= r['hour'] <= 5]
    day_results = [r for r in results if 10 <= r['hour'] <= 16]
    
    if night_results and day_results:
        night_high = sum(1 for r in night_results if r['risk'] == 'High')
        day_high = sum(1 for r in day_results if r['risk'] == 'High')
        print(f"\nTemporal Pattern Check:")
        print(f"  Night hours (20:00-05:00): {night_high}/{len(night_results)} high risk predictions")
        print(f"  Day hours (10:00-16:00): {day_high}/{len(day_results)} high risk predictions")
        if night_high > day_high:
            print(f"  ✓ Night shows higher risk (expected)")
        else:
            print(f"  ⚠️  Day shows higher risk than night (unexpected)")
    
    return results

def compare_with_historical(location):
    """Compare predictions with historical statistics."""
    print("\n" + "=" * 80)
    print(f"HISTORICAL COMPARISON: {location}")
    print("=" * 80)
    
    stats = get_crime_statistics(location)
    if not stats:
        print("No historical data available for this location.")
        return
    
    print(f"\nHistorical Statistics:")
    print(f"  Total crimes: {stats.get('total_crimes', 0):,}")
    print(f"  Most common crime: {stats.get('most_common', 'N/A')}")
    
    crime_freq = stats.get('crime_frequency', {})
    if crime_freq:
        print(f"\nTop Crime Types (Historical):")
        sorted_freq = sorted(crime_freq.items(), key=lambda x: x[1]['count'], reverse=True)
        for crime, info in sorted_freq[:5]:
            print(f"  {crime:20s}: {info['count']:>6,} ({info['percentage']:5.1f}%) - {info['frequency']}")
    
    # Test prediction
    now = datetime.now()
    result = predict_comprehensive(location, now.weekday(), now.hour, now.month, now.year)
    
    print(f"\nCurrent Prediction:")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Predicted Crime Type: {result['predicted_crime_type']}")
    
    # Check if predicted crime type matches historical most common
    most_common_historical = stats.get('most_common')
    predicted_crime = result['predicted_crime_type']
    
    if most_common_historical:
        if predicted_crime == most_common_historical:
            print(f"  ✓ Predicted crime type matches historical most common")
        elif predicted_crime in [c for c, _ in sorted_freq[:3]]:
            print(f"  ✓ Predicted crime type is in top 3 historical crimes")
        else:
            print(f"  ⚠️  Predicted crime type differs from historical most common")
            print(f"     Historical: {most_common_historical}, Predicted: {predicted_crime}")

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTING")
    print("=" * 80)
    
    test_location = "Chicago, IL"
    edge_cases = [
        ("Midnight", 0, 0, 1, 2024),
        ("Noon", 0, 12, 1, 2024),
        ("Late night", 5, 23, 1, 2024),
        ("Monday morning", 0, 8, 1, 2024),
        ("Friday night", 4, 22, 1, 2024),
        ("Saturday night", 5, 23, 1, 2024),
    ]
    
    print("\nTesting edge cases:")
    for name, day, hour, month, year in edge_cases:
        try:
            result = predict_comprehensive(test_location, day, hour, month, year)
            prob_sum = sum(result['risk_probabilities'].values())
            print(f"  {name:20s}: Risk={result['risk_level']:6s}, "
                  f"Crime={result['predicted_crime_type']:15s}, "
                  f"ProbSum={prob_sum:.2%}")
        except Exception as e:
            print(f"  {name:20s}: ERROR - {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_predictions.py <location>")
        print("Example: python validate_predictions.py 'Chicago, IL'")
        sys.exit(1)
    
    location = sys.argv[1]
    
    # Test 1: Compare with historical data
    compare_with_historical(location)
    
    # Test 2: Test consistency across different times
    now = datetime.now()
    test_cases = [
        (0, 14, now.month, now.year),  # Monday afternoon
        (0, 22, now.month, now.year),  # Monday night
        (5, 14, now.month, now.year),  # Saturday afternoon
        (5, 22, now.month, now.year),  # Saturday night
    ]
    validate_prediction_consistency(location, test_cases)
    
    # Test 3: Edge cases
    test_edge_cases()
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nTips for verifying predictions:")
    print("1. Check that risk probabilities sum to ~100%")
    print("2. Verify predicted risk has the highest probability")
    print("3. Compare night vs day predictions (night should generally be higher risk)")
    print("4. Check if predicted crime type aligns with historical data")
    print("5. Test multiple scenarios for the same location")
    print("6. Review model accuracy metrics in outputs/results.txt")

