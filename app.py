"""
Crime Risk Prediction Web Application
Streamlit-based interactive web app for predicting crime risk levels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import folium
from streamlit_folium import st_folium
import time

# Page configuration
st.set_page_config(
    page_title="Crime Risk Prediction",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .risk-high {
        background: linear-gradient(135deg, #ff4b4b 0%, #c41e3a 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #00c853 0%, #009624 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Cache for geocoding results
@st.cache_data(ttl=3600)
def geocode_location(location_name):
    """Geocode a location name to latitude and longitude."""
    try:
        geolocator = Nominatim(user_agent="crime_risk_prediction_streamlit")
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors(model_type='random_forest'):
    """Load trained model and preprocessing components."""
    model_dir = 'outputs'
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    scaler = None
    if model_type == 'logistic_regression' and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler, label_encoder

def create_features(latitude, longitude, day_of_week, hour, month, year):
    """Create feature vector from inputs."""
    grid_precision = 2
    grid_lat = round(latitude, grid_precision)
    grid_lon = round(longitude, grid_precision)
    
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'year': [year],
        'grid_lat': [grid_lat],
        'grid_lon': [grid_lon],
        'past_crime_count': [0],
        'crime_count_30d': [0]
    })
    
    return features

def predict_risk(model, scaler, label_encoder, features):
    """Make prediction using the model."""
    if scaler is not None:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
    else:
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
    
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    class_names = label_encoder.classes_
    probabilities_dict = dict(zip(class_names, probabilities))
    
    return predicted_class, probabilities_dict

def create_probability_chart(probabilities):
    """Create a bar chart for probabilities."""
    colors = {'Low': '#00c853', 'Medium': '#ffa500', 'High': '#ff4b4b'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=[p * 100 for p in probabilities.values()],
            marker_color=[colors.get(k, '#666') for k in probabilities.keys()],
            text=[f'{p*100:.1f}%' for p in probabilities.values()],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Risk Level Probability Distribution",
        xaxis_title="Risk Level",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_hourly_risk_chart(model, scaler, label_encoder, latitude, longitude, day_of_week, month, year):
    """Create a line chart showing risk throughout the day."""
    hours = list(range(24))
    risks = []
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    
    for hour in hours:
        features = create_features(latitude, longitude, day_of_week, hour, month, year)
        predicted_class, _ = predict_risk(model, scaler, label_encoder, features)
        risks.append(risk_mapping.get(predicted_class, 0))
    
    # Create color gradient based on risk
    colors = ['#00c853' if r == 1 else '#ffa500' if r == 2 else '#ff4b4b' for r in risks]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=risks,
        mode='lines+markers',
        line=dict(width=3, color='#667eea'),
        marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title="Risk Level Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Risk Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        ),
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_weekly_heatmap(model, scaler, label_encoder, latitude, longitude, month, year):
    """Create a heatmap showing risk by day and hour."""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    
    data = []
    for day_idx, day in enumerate(days):
        row = []
        for hour in hours:
            features = create_features(latitude, longitude, day_idx, hour, month, year)
            predicted_class, _ = predict_risk(model, scaler, label_encoder, features)
            row.append(risk_mapping.get(predicted_class, 0))
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=hours,
        y=days,
        colorscale=[[0, '#00c853'], [0.5, '#ffa500'], [1, '#ff4b4b']],
        showscale=True,
        colorbar=dict(
            title="Risk",
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        )
    ))
    
    fig.update_layout(
        title="Weekly Risk Heatmap (Hour vs Day)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_dark",
        height=400,
        margin=dict(t=50, b=50, l=100, r=50)
    )
    
    return fig

def create_date_range_chart(model, scaler, label_encoder, latitude, longitude, start_date, end_date):
    """Create a chart showing risk trends over a date range."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    
    data = []
    for date in dates:
        # Calculate average risk for each day (sample a few hours)
        daily_risks = []
        for hour in [0, 6, 12, 18]:  # Sample 4 hours per day
            features = create_features(
                latitude, longitude, 
                date.dayofweek, hour, 
                date.month, date.year
            )
            predicted_class, _ = predict_risk(model, scaler, label_encoder, features)
            daily_risks.append(risk_mapping.get(predicted_class, 0))
        
        avg_risk = np.mean(daily_risks)
        data.append({
            'date': date,
            'risk': avg_risk,
            'risk_label': 'Low' if avg_risk < 1.5 else 'Medium' if avg_risk < 2.5 else 'High'
        })
    
    df = pd.DataFrame(data)
    
    colors = df['risk_label'].map({'Low': '#00c853', 'Medium': '#ffa500', 'High': '#ff4b4b'})
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['risk'],
        mode='lines+markers',
        line=dict(width=2, color='#667eea'),
        marker=dict(size=8, color=colors.tolist()),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Risk Trend: {start_date.strftime('%b %d')} to {end_date.strftime('%b %d, %Y')}",
        xaxis_title="Date",
        yaxis_title="Average Risk Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        ),
        template="plotly_dark",
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig, df

def create_map(latitude, longitude, risk_level, location_name):
    """Create an interactive map with the location marker."""
    # Color based on risk
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    color = colors.get(risk_level, 'blue')
    
    m = folium.Map(location=[latitude, longitude], zoom_start=12, tiles='CartoDB dark_matter')
    
    folium.Marker(
        [latitude, longitude],
        popup=f"<b>{location_name}</b><br>Risk: {risk_level}",
        tooltip=f"Risk: {risk_level}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)
    
    # Add a circle to show the area
    folium.Circle(
        [latitude, longitude],
        radius=1000,
        color=color,
        fill=True,
        fill_opacity=0.2
    ).add_to(m)
    
    return m

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 Crime Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict crime risk levels for any location and time</p>', unsafe_allow_html=True)
    
    # Load model
    model_type = st.sidebar.selectbox(
        "🤖 Select Model",
        ["random_forest", "logistic_regression"],
        format_func=lambda x: "Random Forest (Recommended)" if x == "random_forest" else "Logistic Regression"
    )
    
    model, scaler, label_encoder = load_model_and_preprocessors(model_type)
    
    if model is None:
        st.error("⚠️ Model not found! Please ensure the model files are in the 'outputs' directory.")
        st.info("Run `python src/04_train_eval.py` to train the models first.")
        return
    
    # Sidebar inputs
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📍 Location")
    
    # Location input
    location_input = st.sidebar.text_input(
        "Enter location name",
        placeholder="e.g., Chicago, IL or New York, NY",
        help="Enter a city name, address, or any location"
    )
    
    # Geocode button
    lat, lon, address = None, None, None
    if st.sidebar.button("🔍 Search Location", use_container_width=True):
        if location_input:
            with st.spinner("Searching location..."):
                lat, lon, address = geocode_location(location_input)
                if lat and lon:
                    st.session_state['latitude'] = lat
                    st.session_state['longitude'] = lon
                    st.session_state['address'] = address
                    st.sidebar.success(f"Found: {address}")
                else:
                    st.sidebar.error("Location not found. Try a different search term.")
    
    # Use stored location or manual input
    if 'latitude' in st.session_state:
        lat = st.session_state['latitude']
        lon = st.session_state['longitude']
        address = st.session_state.get('address', '')
    
    # Manual coordinate input
    st.sidebar.markdown("### Or enter coordinates manually:")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        manual_lat = st.number_input("Latitude", value=lat if lat else 41.8781, format="%.4f")
    with col2:
        manual_lon = st.number_input("Longitude", value=lon if lon else -87.6298, format="%.4f")
    
    # Use manual input if no geocoded location
    if lat is None:
        lat = manual_lat
        lon = manual_lon
    
    # Date and Time inputs
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📅 Date & Time")
    
    selected_date = st.sidebar.date_input("Select Date", datetime.now())
    selected_hour = st.sidebar.slider("Select Hour", 0, 23, datetime.now().hour, 
                                       format="%d:00",
                                       help="0 = Midnight, 12 = Noon, 23 = 11 PM")
    
    day_of_week = selected_date.weekday()
    month = selected_date.month
    year = selected_date.year
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    st.sidebar.info(f"📆 {days[day_of_week]}, {selected_date.strftime('%B %d, %Y')} at {selected_hour}:00")
    
    # Analysis mode
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose analysis type:",
        ["Single Prediction", "Date Range Analysis", "Weekly Heatmap"]
    )
    
    # Date range inputs
    if analysis_mode == "Date Range Analysis":
        st.sidebar.markdown("### Select Date Range:")
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now(), datetime.now() + timedelta(days=7)),
            help="Select start and end dates for trend analysis"
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0]
    
    # Main content area
    st.markdown("---")
    
    # Make prediction
    features = create_features(lat, lon, day_of_week, selected_hour, month, year)
    predicted_class, probabilities = predict_risk(model, scaler, label_encoder, features)
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Prediction Result")
        
        # Risk level display
        risk_class = f"risk-{predicted_class.lower()}"
        st.markdown(f'<div class="{risk_class}">⚠️ {predicted_class.upper()} RISK</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Location info
        st.markdown(f"""
        **📍 Location:** {address if address else f'({lat:.4f}, {lon:.4f})'}  
        **📅 Date:** {days[day_of_week]}, {selected_date.strftime('%B %d, %Y')}  
        **🕐 Time:** {selected_hour}:00  
        **🤖 Model:** {model_type.replace('_', ' ').title()}
        """)
        
        # Probability chart
        prob_chart = create_probability_chart(probabilities)
        st.plotly_chart(prob_chart, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Location Map")
        
        # Create and display map
        risk_map = create_map(lat, lon, predicted_class, address if address else f'({lat:.4f}, {lon:.4f})')
        st_folium(risk_map, width=None, height=400)
    
    # Additional analysis based on mode
    st.markdown("---")
    
    if analysis_mode == "Single Prediction":
        st.markdown("### 📈 Risk Throughout the Day")
        hourly_chart = create_hourly_risk_chart(model, scaler, label_encoder, lat, lon, day_of_week, month, year)
        st.plotly_chart(hourly_chart, use_container_width=True)
        
    elif analysis_mode == "Date Range Analysis":
        st.markdown("### 📈 Risk Trend Analysis")
        if len(date_range) == 2:
            trend_chart, trend_data = create_date_range_chart(model, scaler, label_encoder, lat, lon, start_date, end_date)
            st.plotly_chart(trend_chart, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                low_days = len(trend_data[trend_data['risk_label'] == 'Low'])
                st.metric("Low Risk Days", low_days, delta=None)
            with col2:
                med_days = len(trend_data[trend_data['risk_label'] == 'Medium'])
                st.metric("Medium Risk Days", med_days, delta=None)
            with col3:
                high_days = len(trend_data[trend_data['risk_label'] == 'High'])
                st.metric("High Risk Days", high_days, delta=None)
        
    elif analysis_mode == "Weekly Heatmap":
        st.markdown("### 🗓️ Weekly Risk Heatmap")
        heatmap = create_weekly_heatmap(model, scaler, label_encoder, lat, lon, month, year)
        st.plotly_chart(heatmap, use_container_width=True)
        st.info("💡 This heatmap shows risk levels for every hour of each day of the week. "
                "Use this to identify the safest times to visit this location.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚨 Crime Risk Prediction System | Built with Streamlit</p>
        <p>⚠️ This is a prediction tool based on historical data. Actual crime risk may vary.</p>
        <p style='margin-top: 15px; font-size: 0.9rem;'>© 2025 Vemireddy. All Rights Reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

