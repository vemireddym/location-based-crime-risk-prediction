import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import folium
from streamlit_folium import st_folium

sys.path.append('src')
try:
    from predict import predict_comprehensive, get_crime_statistics
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict", "src/predict.py")
    predict_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_module)
    predict_comprehensive = predict_module.predict_comprehensive
    get_crime_statistics = predict_module.get_crime_statistics

st.set_page_config(
    page_title="Crime Risk Prediction",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def geocode_location(location_name):
    try:
        geolocator = Nominatim(user_agent="crime_risk_prediction_streamlit")
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except:
        return None, None, None

def create_crime_frequency_chart(crime_stats):
    if not crime_stats or 'crime_frequency' not in crime_stats:
        return None
    
    crime_freq = crime_stats['crime_frequency']
    if not crime_freq:
        return None
    
    top_crimes = sorted(crime_freq.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    
    crimes = [c[0] for c in top_crimes]
    counts = [c[1]['count'] for c in top_crimes]
    frequencies = [c[1]['frequency'] for c in top_crimes]
    
    colors = []
    for freq in frequencies:
        if freq == 'Very Common':
            colors.append('#ff4b4b')
        elif freq == 'Common':
            colors.append('#ffa500')
        elif freq == 'Occasional':
            colors.append('#ffd700')
        else:
            colors.append('#00c853')
    
    fig = go.Figure(data=[
        go.Bar(
            x=crimes,
            y=counts,
            marker_color=colors,
            text=[f"{c} ({f})" for c, f in zip(counts, frequencies)],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Top Crime Types by Frequency",
        xaxis_title="Crime Type",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_map(lat, lon, risk_level, location_name):
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    color = colors.get(risk_level, 'blue')
    
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles='CartoDB dark_matter')
    folium.Marker(
        [lat, lon],
        popup=f"<b>{location_name}</b><br>Risk: {risk_level}",
        tooltip=f"Risk: {risk_level}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)
    
    return m

def main():
    st.markdown('<h1 class="main-header">🚨 Crime Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict crime risk levels and types for any location and time</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📍 Location")
    location_input = st.sidebar.text_input(
        "Enter location (City, State)",
        placeholder="e.g., Chicago, IL",
        help="Enter city and state"
    )
    
    lat, lon, address = None, None, None
    if st.sidebar.button("🔍 Get Coordinates", use_container_width=True):
        if location_input:
            with st.spinner("Searching..."):
                lat, lon, address = geocode_location(location_input)
                if lat and lon:
                    st.session_state['location'] = location_input
                    st.session_state['lat'] = lat
                    st.session_state['lon'] = lon
                    st.sidebar.success(f"Found: {address}")
                else:
                    st.sidebar.error("Location not found")
    
    if 'location' in st.session_state:
        location = st.session_state['location']
        lat = st.session_state.get('lat')
        lon = st.session_state.get('lon')
    else:
        location = location_input if location_input else None
    
    st.sidebar.markdown("## 📅 Date & Time")
    selected_date = st.sidebar.date_input("Select Date", datetime.now())
    selected_hour = st.sidebar.slider("Select Hour", 0, 23, datetime.now().hour, format="%d:00")
    
    day_of_week = selected_date.weekday()
    month = selected_date.month
    year = selected_date.year
    
    if not location:
        st.info("👈 Enter a location in the sidebar to get predictions")
        return
    
    try:
        result = predict_comprehensive(location, day_of_week, selected_hour, month, year)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 Risk Level Prediction")
            risk_class = f"risk-{result['risk_level'].lower()}"
            st.markdown(f'<div class="{risk_class}">⚠️ {result["risk_level"].upper()} RISK</div>', unsafe_allow_html=True)
            
            risk_proba = result['risk_probabilities']
            fig_risk = go.Figure(data=[
                go.Bar(
                    x=list(risk_proba.keys()),
                    y=[p * 100 for p in risk_proba.values()],
                    marker_color=['#00c853', '#ffa500', '#ff4b4b'],
                    text=[f'{p*100:.1f}%' for p in risk_proba.values()],
                    textposition='outside'
                )
            ])
            fig_risk.update_layout(
                title="Risk Level Probabilities",
                yaxis_title="Probability (%)",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Crime Type Prediction")
            crime_type = result['predicted_crime_type']
            crime_prob = result['crime_type_probability'] * 100
            
            st.metric("Predicted Crime Type", crime_type, f"{crime_prob:.1f}% confidence")
            
            if result.get('crime_statistics'):
                stats = result['crime_statistics']
                if 'total_crimes' in stats:
                    st.metric("Total Crimes (Historical)", f"{stats['total_crimes']:,}")
                if 'most_common' in stats and stats['most_common']:
                    st.metric("Most Common Crime Type", stats['most_common'])
        
        st.markdown("---")
        st.markdown("### 📊 Crime Frequency Statistics")
        
        if result.get('crime_statistics') and result['crime_statistics'].get('crime_frequency'):
            crime_freq = result['crime_statistics']['crime_frequency']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                freq_chart = create_crime_frequency_chart(result['crime_statistics'])
                if freq_chart:
                    st.plotly_chart(freq_chart, use_container_width=True)
            
            with col2:
                st.markdown("#### Frequency Legend")
                st.markdown("🔴 **Very Common**: >30%")
                st.markdown("🟠 **Common**: 15-30%")
                st.markdown("🟡 **Occasional**: 5-15%")
                st.markdown("🟢 **Rare**: <5%")
                
                st.markdown("#### Top Crime Types")
                sorted_crimes = sorted(crime_freq.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
                for crime, data in sorted_crimes:
                    st.write(f"**{crime}**")
                    st.write(f"  Count: {data['count']:,}")
                    st.write(f"  Frequency: {data['frequency']}")
                    st.write("")
        
        if lat and lon:
            st.markdown("---")
            st.markdown("### 🗺️ Location Map")
            risk_map = create_map(lat, lon, result['risk_level'], location)
            st_folium(risk_map, width=None, height=400)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Make sure the models are trained. Run the training pipeline first.")
    
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
