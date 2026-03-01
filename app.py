import streamlit as st

st.set_page_config(
    page_title="2026 F1 Prediction Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏎️ F1 Prediction 3000")
    
    st.markdown("""
    Welcome to the **Unified 2026 Formula 1 Race Predictor**! 
    
    This application replaces the 12 individual prediction scripts with a single, massive Gradient Boosting machine learning engine. 
    It uses:
    * **2024 & 2025 Historical Data** (Sector & Lap times via FastF1)
    * **2026 Pre-season Testing Data** (Barcelona & Bahrain)
    * **Live Weather Data** (via OpenWeatherMap)
    
    ### 👈 Select a page from the sidebar to get started
    
    * **🏁 Race Predictor**: Select any grand prix on the 2026 calendar and run the prediction model to see the forecasted grid order.
    * **📊 Season Overview**: Review pre-season testing results, pacing, and basic driver statistics.
    """)
    
    st.info("The prediction engine pulls thousands of data points from the FastF1 cache. The very first prediction run may take 10-20 seconds as historical data is aggregated. Subsequent runs will be much faster.")
    
    # Custom CSS for F1 branding Look and Feel
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        h1, h2, h3 {
            color: #E8002D;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            background-color: #E8002D;
            color: white;
            border-radius: 4px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #FF1A46;
            color: white;
            border-color: #FF1A46;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
