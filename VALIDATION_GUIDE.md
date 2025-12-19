# How to Validate Prediction Accuracy

This guide explains multiple ways to verify that your crime risk predictions are correct and reliable.

## Quick Validation Checklist

### 1. ✅ Check Probability Consistency
- **Risk probabilities should sum to ~100%** (within 0.01 tolerance)
- **Predicted risk level should have the highest probability**
- Example: If predicted risk is "High", then `P(High) > P(Medium)` and `P(High) > P(Low)`

### 2. ✅ Compare with Historical Data
The prediction output already shows historical statistics. Check:
- Does the predicted crime type match or align with historical most common crimes?
- Does the risk level make sense given the location's historical crime patterns?
- Are the crime type probabilities reasonable compared to historical frequencies?

### 3. ✅ Test Temporal Patterns
Run predictions for the same location at different times:
- **Night hours (20:00-05:00)** should generally show **higher risk** than day hours
- **Weekend nights** might differ from weekday nights
- **Late night (22:00-02:00)** typically has higher risk than afternoon (12:00-16:00)

### 4. ✅ Test Multiple Scenarios
Test the same location with different inputs:
```bash
# Monday afternoon
python src/05_predict.py "Chicago, IL" 0 14 3 2024

# Monday night
python src/05_predict.py "Chicago, IL" 0 22 3 2024

# Saturday night
python src/05_predict.py "Chicago, IL" 5 23 3 2024
```

Compare the results - they should show logical differences based on time/day.

## Using the Validation Script

A validation script is provided to automate these checks:

```bash
python src/validate_predictions.py "Chicago, IL"
```

This script will:
1. Compare predictions with historical statistics
2. Test consistency across multiple scenarios
3. Check probability sums and prediction alignment
4. Test edge cases (midnight, noon, weekend nights, etc.)
5. Validate temporal patterns (night vs day)

## Manual Validation Steps

### Step 1: Check Model Accuracy Metrics
```bash
cat outputs/results.txt
```

Look for:
- **Risk Model Accuracy**: Should be > 90% (your model shows 99.92%)
- **Crime Type Model Accuracy**: Should be > 90% (your model shows 99.99%)
- **F1-Scores**: Should be > 0.90

### Step 2: Verify Historical Alignment
When you run a prediction, check:
1. **Predicted Crime Type** vs **Historical Most Common**
   - Should match or be in top 3 historical crimes
2. **Risk Level** vs **Historical Crime Count**
   - High crime locations should predict higher risk
   - Low crime locations should predict lower risk

### Step 3: Test Known Scenarios
Test locations and times where you have expectations:
- **High-crime area at night** → Should predict High/Medium risk
- **Low-crime area during day** → Should predict Low risk
- **Weekend night** → Might differ from weekday night

### Step 4: Check Probability Distributions
For each prediction, verify:
- Risk probabilities: `P(Low) + P(Medium) + P(High) ≈ 1.0`
- Crime type probabilities: Should sum to 1.0 (if all shown)
- Predicted class should have maximum probability

### Step 5: Cross-Validation
Run multiple predictions for the same location:
- Same location, different times → Results should vary logically
- Same time, different days → Weekend vs weekday differences
- Same location, different months → Seasonal patterns (if any)

## Expected Behaviors

### ✅ Good Signs (Predictions are likely correct):
- Probabilities sum to ~100%
- Predicted risk has highest probability
- Night predictions show higher risk than day
- Predicted crime type aligns with historical data
- Different times show logical variations
- Model accuracy metrics are high (>90%)

### ⚠️ Warning Signs (May indicate issues):
- Probabilities don't sum to 100%
- Predicted risk doesn't have highest probability
- All predictions show same risk level regardless of time
- Predicted crime type never matches historical data
- Night predictions show lower risk than day (unexpected)

## Model Performance Benchmarks

Based on your training results:
- **Risk Level Model**: 99.92% accuracy, 99.91% F1-score
- **Crime Type Model**: 99.99% accuracy, 96.75% F1-score

These are excellent metrics, so predictions should be highly reliable.

## Example Validation Test

```bash
# Test 1: Check historical alignment
python src/05_predict.py "Chicago, IL" 0 14 3 2024

# Test 2: Compare night vs day
python src/05_predict.py "Chicago, IL" 0 14 3 2024  # Day
python src/05_predict.py "Chicago, IL" 0 22 3 2024  # Night

# Test 3: Compare weekday vs weekend
python src/05_predict.py "Chicago, IL" 0 22 3 2024  # Monday night
python src/05_predict.py "Chicago, IL" 5 22 3 2024  # Saturday night

# Test 4: Use validation script
python src/validate_predictions.py "Chicago, IL"
```

## Interpreting Results

### High Confidence Predictions:
- Risk probability > 50% for predicted class
- Crime type probability > 40%
- Aligns with historical patterns

### Medium Confidence Predictions:
- Risk probability 30-50% for predicted class
- Crime type probability 20-40%
- Some alignment with historical patterns

### Low Confidence Predictions:
- Risk probability < 30% for predicted class
- Crime type probability < 20%
- Doesn't align with historical patterns
- **Action**: May need to retrain model or check input data

## Troubleshooting

If predictions seem incorrect:

1. **Check model files**: Ensure `outputs/` contains valid model files
2. **Verify training data**: Check if location exists in training data
3. **Check feature creation**: Ensure features are created correctly
4. **Review model metrics**: Check `outputs/results.txt` for accuracy
5. **Retrain if needed**: Run `python src/04_train_eval.py` to retrain

## Summary

The best way to validate predictions is to:
1. ✅ Use the validation script for automated checks
2. ✅ Compare predictions with historical statistics (shown in output)
3. ✅ Test multiple scenarios and check for logical patterns
4. ✅ Verify probability consistency
5. ✅ Review model accuracy metrics

Your model has very high accuracy (99%+), so predictions should be reliable when:
- Input location exists in training data
- Model files are not corrupted
- Features are created correctly

