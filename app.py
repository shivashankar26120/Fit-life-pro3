import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from fitness_engine import FitnessEngine
from auth_system import AuthSystem

# Page Configuration
st.set_page_config(
    page_title="FitLife Pro - AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better visibility
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main background with better contrast */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
    }

    /* Make all text visible */
    .stMarkdown, .stText, p, span, div {
        color: #ffffff !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.75rem 2rem;
        font-size: 16px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }

    /* Card styling with better contrast */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin: 1rem 0;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box_shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    .metric-card h3, .metric-card h4, .metric-card p, .metric-card span {
        color: #1e3c72 !important;
    }

    .metric-card h1, .metric-card h2 {
        color: #667eea !important;
    }

    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .success-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .warning-box {
        background: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Meal card styling */
    .meal-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }

    .meal-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    }

    .meal-card h4, .meal-card h5, .meal-card p, .meal-card span {
        color: #1e3c72 !important;
    }

    /* Workout card */
    .workout-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }

    .workout-card h4, .workout-card p, .workout-card span {
        color: #1e3c72 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: white !important;
        font-weight: 600;
        padding: 10px 20px;
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }

    .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }

    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: rgba(255,255,255,0.9);
        color: #1e3c72 !important;
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 0.75rem;
    }

    .stSelectbox>div>div>div {
        background-color: rgba(255,255,255,0.9);
        color: #1e3c72 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.1);
        border-radius: 8px;
        color: white !important;
        font-weight: 600;
    }

    .streamlit-expanderContent {
        background-color: rgba(255,255,255,0.05);
        border-radius: 0 0 8px 8px;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }

    /* Remove padding on mobile */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
    }

    /* Login container */
    .login-container {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        max-width: 450px;
        margin: 3rem auto;
    }

    .login-container h1, .login-container h2, .login-container h3,
    .login-container p, .login-container span, .login-container label {
        color: #1e3c72 !important;
    }

    /* Stats display */
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea !important;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #666 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }に対し
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Dataframe styling */
    .dataframe {
        background-color: white !important;
    }

    .dataframe th {
        background-color: #667eea !important;
        color: white !important;
    }

    .dataframe td {
        color: #1e3c72 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
    if 'plan_generated' not in st.session_state:
        st.session_state.plan_generated = False
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'fitness_engine' not in st.session_state:
        st.session_state.fitness_engine = None
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = AuthSystem()

init_session_state()

# Load models (cached)
@st.cache_resource
def load_fitness_engine():
    """Load and train the fitness engine"""
    engine = FitnessEngine()
    result = engine.load_and_train_models(
        'health_fitness_dataset_small.csv',
        'Indian_Food_Nutrition_Processed.csv'
    )
    if result['success']:
        return engine, result
    return None, result

# Initialize fitness engine
if st.session_state.fitness_engine is None:
    with st.spinner("🔄 Loading AI models..."):
        engine, result = load_fitness_engine()
        if engine:
            st.session_state.fitness_engine = engine
        else:
            st.error(f"❌ Failed to load models: {result.get('error', 'Unknown error')}")
            st.stop()


# --- User Authentication ---
"""
Login and Registration Pages
"""

def login_page():
    """Display login and registration page"""
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>💪 FitLife Pro</h1>
            <p style='font-size: 1.3rem; opacity: 0.9;'>Your AI-Powered Fitness & Nutrition Coach</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container fade-in">', unsafe_allow_html=True)
        
        # Tabs for Login and Sign Up
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        # LOGIN TAB
        with tab1:
            st.markdown("### Welcome Back!")
            st.markdown("---")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_btn = st.form_submit_button("Login", use_container_width=True)
                with col_b:
                    forgot_btn = st.form_submit_button("Forgot Password?", use_container_width=True)
                
                if login_btn:
                    if not username or not password:
                        st.error("❌ Please fill in all fields")
                    else:
                        result = st.session_state.auth_system.login_user(username, password)
                        
                        if result['success']:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_profile = result['user_data']['profile']
                            st.success("✅ Login successful!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                
                if forgot_btn:
                    st.info("💡 Please contact admin to reset your password")
        
        # SIGN UP TAB
        with tab2:
            st.markdown("### Create Your Account")
            st.markdown("---")
            
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="Enter username (min 4 characters)")
                email = st.text_input("Email (Optional)", placeholder="your.email@example.com")
                new_password = st.text_input("Choose Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                
                agree = st.checkbox("I agree to Terms & Conditions")
                
                signup_btn = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_btn:
                    if not new_username or not new_password or not confirm_password:
                        st.error("❌ Please fill in all required fields")
                    elif len(new_username) < 4:
                        st.error("❌ Username must be at least 4 characters")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords don't match!")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    elif not agree:
                        st.error("❌ Please agree to Terms & Conditions")
                    else:
                        result = st.session_state.auth_system.register_user(
                            new_username, 
                            new_password,
                            email
                        )
                        
                        if result['success']:
                            st.success("✅ Account created successfully! Please login.")
                            st.balloons()
                        else:
                            st.error(f"❌ {result['message']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("🤖", "AI-Powered", "Machine Learning recommendations"),
        ("🎯", "Personalized", "Tailored to your goals"),
        ("📊", "Progress Tracking", "Monitor your journey"),
        ("🍽️", "Smart Meals", "Indian cuisine database")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
                    <h3 style="margin: 0.5rem 0;">{title}</h3>
                    <p style="font-size: 0.9rem; opacity: 0.8; margin: 0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 2rem; margin-top: 3rem; opacity: 0.8;'>
            <p>Built with ❤️ using Machine Learning & Streamlit</p>
            <p style='font-size: 0.9rem;'>© 2024 FitLife Pro. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)
    """
User Profile Setup and Main Dashboard
"""

def profile_setup_sidebar():
    """Sidebar for user profile and settings"""
    
    with st.sidebar:
        # User info header
        st.markdown(f"""
            <div class="info-box">
                <h3 style="margin: 0;">👤 {st.session_state.username}</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">Member since {datetime.now().strftime('%B %Y')}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Profile section
        st.markdown("### 📋 Your Profile")
        
        # Get current profile
        profile = st.session_state.user_profile
        
        # Personal Info
        st.markdown("#### Basic Information")
        age = st.number_input("Age", 15, 100, int(profile.get('age', 25)) if profile.get('age') else 25, key="age")
        gender = st.selectbox("Gender", ["Male", "Female"], 
                             index=0 if profile.get('gender') in ['Male', 'M'] else 1, 
                             key="gender")
        height = st.number_input("Height (cm)", 100, 250, 
                                int(profile.get('height_cm', 170)) if profile.get('height_cm') else 170,
                                key="height")
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 
                                float(profile.get('weight_kg', 70)) if profile.get('weight_kg') else 70.0,
                                step=0.1, key="weight")
        
        st.markdown("---")
        
        # Activity Info
        st.markdown("#### Activity Level")
        daily_steps = st.slider("Average Daily Steps", 1000, 20000, 
                               int(profile.get('daily_steps', 8000)) if profile.get('daily_steps') else 8000,
                               500, key="steps")
        
        fitness_level_name = st.select_slider(
            "Current Fitness Level",
            options=["Beginner", "Intermediate", "Advanced"],
            value=profile.get('fitness_level_name', 'Intermediate') if profile.get('fitness_level_name') else 'Intermediate',
            key="fitness_level"
        )
        
        st.markdown("---")
        
        # Goal Setting
        st.markdown("#### 🎯 Your Goals")
        goal = st.selectbox(
            "Primary Goal",
            ["Weight Loss", "Weight Gain", "Muscle Building", "Maintenance"],
            index=["Weight Loss", "Weight Gain", "Muscle Building", "Maintenance"].index(
                profile.get('goal', 'Maintenance') if profile.get('goal') in ["Weight Loss", "Weight Gain", "Muscle Building", "Maintenance"] else 'Maintenance'
            ),
            key="goal"
        )
        
        target_days = st.number_input(
            "Goal Duration (days)", 
            7, 365, 
            int(profile.get('target_days', 30)) if profile.get('target_days') else 30,
            7,
            help="How many days do you want to follow this plan?",
            key="target_days"
        )
        
        # Weight change target (NEW FEATURE)
        if goal == "Weight Loss":
            target_weight_change = st.number_input(
                "Target Weight Loss (kg)",
                0.5, 30.0, 
                float(profile.get('target_weight_change', 5.0)) if profile.get('target_weight_change') else 5.0,
                0.5,
                help="How much weight do you want to lose?",
                key="target_weight_change"
            )
        elif goal == "Weight Gain":
            target_weight_change = st.number_input(
                "Target Weight Gain (kg)",
                0.5, 20.0,
                float(profile.get('target_weight_change', 3.0)) if profile.get('target_weight_change') else 3.0,
                0.5,
                help="How much weight do you want to gain?",
                key="target_weight_change"
            )
        else:
            target_weight_change = 0
        
        st.markdown("---")
        
        # Generate Plan Button
        if st.button("🚀 Generate My Plan", use_container_width=True, type="primary"):
            generate_plan(age, gender, height, weight, daily_steps, fitness_level_name, 
                         goal, target_days, target_weight_change)
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()

def generate_plan(age, gender, height, weight, daily_steps, fitness_level_name, 
                  goal, target_days, target_weight_change):
    """Generate personalized fitness and nutrition plan"""
    
    with st.spinner("🤖 AI is analyzing your profile and creating your personalized plan..."):
        # Save profile
        fitness_map = {"Beginner": 0.2, "Intermediate": 0.5, "Advanced": 0.8}
        fitness_level = fitness_map[fitness_level_name]
        
        profile_data = {
            'age': age,
            'gender': gender,
            'height_cm': height,
            'weight_kg': weight,
            'daily_steps': daily_steps,
            'fitness_level': fitness_level,
            'fitness_level_name': fitness_level_name,
            'goal': goal,
            'target_days': target_days,
            'target_weight_change': target_weight_change
        }
        
        st.session_state.user_profile = profile_data
        st.session_state.auth_system.update_user_profile(
            st.session_state.username,
            profile_data
        )
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Calculate BMR
        gender_code = 'M' if gender == 'Male' else 'F'
        if gender_code == 'M':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        # Activity level
        if daily_steps < 3000:
            activity_mult = 1.2
            activity_level = 'Sedentary'
        elif daily_steps < 7500:
            activity_mult = 1.375
            activity_level = 'Lightly Active'
        elif daily_steps < 10000:
            activity_mult = 1.55
            activity_level = 'Moderately Active'
        else:
            activity_mult = 1.725
            activity_level = 'Very Active'
        
        maintenance_calories = bmr * activity_mult
        
        # Predict workout
        engine = st.session_state.fitness_engine
        user_data = {
            'age': age,
            'bmi': bmi,
            'fitness_level': fitness_level,
            'daily_steps': daily_steps,
            'stress_level': 5
        }
        
        workout_plan = engine.predict_workout(user_data)
        
        # Calculate target calories based on goal and desired weight change
        if goal in ['Weight Loss', 'weight_loss'] and target_weight_change > 0:
            # Calculate daily deficit needed
            total_deficit_needed = target_weight_change * 7700  # calories
            daily_deficit = total_deficit_needed / target_days
            target_calories = maintenance_calories - daily_deficit
            # Ensure safe deficit (max 1000 cal/day)
            target_calories = max(target_calories, maintenance_calories - 1000)
        elif goal in ['Weight Gain', 'weight_gain'] and target_weight_change > 0:
            total_surplus_needed = target_weight_change * 7700
            daily_surplus = total_surplus_needed / target_days
            target_calories = maintenance_calories + daily_surplus
            # Ensure safe surplus (max 500 cal/day)
            target_calories = min(target_calories, maintenance_calories + 500)
        elif goal in ['Muscle Building', 'muscle_building']:
            target_calories = maintenance_calories + 300
        else:
            target_calories = maintenance_calories
        
        # Generate meal plan
        weekly_meals = engine.recommend_smart_meals(goal, target_calories, 7)
        
        # Calculate hydration
        hydration = engine.calculate_hydration(weight, activity_level.lower().replace(' ', '_'))
        
        # Calculate weight projection
        actual_daily_deficit = maintenance_calories - target_calories
        weight_projection = engine.calculate_target_weight(
            weight, goal, target_days, actual_daily_deficit
        )
        
        # Store recommendations
        st.session_state.recommendations = {
            'bmi': bmi,
            'bmr': bmr,
            'activity_level': activity_level,
            'maintenance_calories': maintenance_calories,
            'target_calories': target_calories,
            'workout': workout_plan,
            'meal_plan': weekly_meals,
            'hydration': hydration,
            'weight_projection': weight_projection,
            'goal': goal,
            'target_days': target_days,
            'target_weight_change': target_weight_change
        }
        
        st.session_state.plan_generated = True
        
        # Save meal plan
        st.session_state.auth_system.save_meal_plan(
            st.session_state.username,
            weekly_meals
        )
        
        st.success("✅ Your personalized plan is ready!")
        st.rerun()

def logout_user():
    """Logout user and clear session"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_profile = {}
    st.session_state.plan_generated = False
    st.session_state.recommendations = None
    st.success("👋 Logged out successfully!")
    st.rerun()
    """
Dashboard Display Functions
"""

def display_dashboard():
    """Main dashboard after login"""
    
    # Header
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown(f"""
            <div style="text-align: center;">
                <h1 style="margin: 0;">💪 FitLife Pro</h1>
                <p style="margin: 0.5rem 0; font-size: 1.1rem;">Welcome, {st.session_state.username}!</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.plan_generated and st.session_state.recommendations:
        display_recommendations()
    else:
        display_welcome_screen()

def display_welcome_screen():
    """Welcome screen before plan generation"""
    
    st.markdown("""
        <div class="metric-card" style="text-align: center; padding: 3rem;">
            <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">Welcome to FitLife Pro! 👋</h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                Get started by filling in your profile details in the sidebar
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h3>🎯 What You'll Get</h3>
                <ul style="line-height: 2;">
                    <li><strong>7 Detailed Workout Plans</strong> - Complete weekly schedules</li>
                    <li><strong>Smart Meal Recommendations</strong> - Based on your goals</li>
                    <li><strong>Weight Projections</strong> - See your progress before you start</li>
                    <li><strong>Hydration Tracking</strong> - Stay properly hydrated</li>
                    <li><strong>Progress Monitoring</strong> - Track your journey</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="success-box">
                <h3>✨ How It Works</h3>
                <ol style="line-height: 2;">
                    <li><strong>Enter Your Details</strong> - Age, weight, height, activity level</li>
                    <li><strong>Set Your Goal</strong> - Weight loss, gain, or maintenance</li>
                    <li><strong>Set Target</strong> - How much weight and in how many days</li>
                    <li><strong>Get Your Plan</strong> - AI creates a personalized program</li>
                    <li><strong>Track Progress</strong> - Monitor and adapt as you go</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    stats = [
        ("🤖", "AI-Powered", "95%+ Accuracy"),
        ("🍽️", "1000+ Dishes", "Indian Cuisine"),
        ("💪", "7 Workout Types", "All fitness levels"),
        ("📊", "Smart Tracking", "Monitor progress")
    ]
    
    for col, (icon, title, value) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <h4 style="margin: 0.5rem 0; font-size: 1rem;">{title}</h4>
                    <p style="font-size: 1.1rem; font-weight: 600; margin: 0;">{value}</p>
                </div>
            """, unsafe_allow_html=True)

def display_recommendations():
    """Display generated recommendations"""
    
    rec = st.session_state.recommendations
    
    # Success message
    st.markdown(f"""
        <div class="success-box" style="text-align: center; padding: 1.5rem;">
            <h2 style="margin: 0;">🎉 Your {rec['target_days']}-Day Personalized Plan is Ready!</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Goal: <strong>{rec['goal']}</strong> | 
                Target: <strong>{rec['target_weight_change']} kg</strong> in <strong>{rec['target_days']} days</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics
    display_key_metrics(rec)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💪 Workout Plans",
        "🍽️ Meal Plan",
        "📊 Progress Tracking",
        "💧 Hydration",
        "📝 Log Progress"
    ])
    
    with tab1:
        display_workout_plans(rec)
    
    with tab2:
        display_meal_plans(rec)
    
    with tab3:
        display_progress_tracking(rec)
    
    with tab4:
        display_hydration_plan(rec)
    
    with tab5:
        display_progress_logging()

def display_key_metrics(rec):
    """Display key health metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("BMI", f"{rec['bmi']:.1f}", "Body Mass Index", get_bmi_status(rec['bmi'])),
        ("BMR", f"{int(rec['bmr'])}", "Calories/day at rest", "kcal"),
        ("Target Calories", f"{int(rec['target_calories'])}", "Daily intake goal", "kcal"),
        ("Water", f"{rec['hydration']['daily_water_liters']}L", "Daily hydration", "goal")
    ]
    
    for col, (title, value, subtitle, unit) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">{title}</h4>
                    <div class="stat-number">{value}</div>
                    <p style="margin: 0; font-size: 0.85rem;">{subtitle}</p>
                    <span style="font-size: 0.8rem; opacity: 0.7;">{unit}</span>
                </div>
            """, unsafe_allow_html=True)

def get_bmi_status(bmi):
    """Get BMI status message"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"
        """
Workout Plans Display Section
"""

def display_workout_plans(rec):
    """Display detailed workout plans with alternatives"""
    
    workout = rec['workout']
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Primary Workout Plan
    st.markdown(f"""
        <div class="info-box">
            <h2 style="margin: 0;">🏆 Your Primary Workout Plan</h2>
            <h3 style="margin: 0.5rem 0; color: #667eea !important;">{workout['primary']['name']}</h3>
            <p style="margin: 0; font-size: 1rem;">
                AI Confidence: <strong>{workout['primary']['confidence']:.1f}%</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display weekly schedule
    primary_details = workout['primary']['details']
    
    if primary_details and 'weekly_plan' in primary_details:
        st.markdown("### 📅 Your Weekly Schedule")
        
        for day_plan in primary_details['weekly_plan']:
            with st.expander(f"**{day_plan['day']}** - {day_plan['type']}", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Activity:** {day_plan['type']}")
                    if 'exercises' in day_plan:
                        st.markdown(f"**Exercises:** {day_plan['exercises']}")
                
                with col2:
                    st.markdown(f"**Duration:** {day_plan['duration']}")
                
                with col3:
                    intensity_color = {
                        'Low': '#4caf50',
                        'Medium': '#ff9800',
                        'High': '#f44336',
                        'Rest': '#9e9e9e'
                    }
                    color = intensity_color.get(day_plan['intensity'], '#667eea')
                    st.markdown(f"**Intensity:** <span style='color: {color}; font-weight: bold;'>{day_plan['intensity']}</span>", 
                               unsafe_allow_html=True)
                
                # Tips for the day
                if day_plan['intensity'] == 'High':
                    st.info("💡 **Tip:** Ensure proper warm-up and cool-down. Stay hydrated!")
                elif day_plan['intensity'] == 'Rest':
                    st.info("💡 **Tip:** Use this day for light stretching or complete rest.")
        
        # Workout summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="workout-card">
                <h4>📊 Weekly Summary</h4>
                <p><strong>Frequency:</strong> {primary_details.get('frequency', '4-5 days/week')}</p>
                <p><strong>Rest Days:</strong> {sum(1 for d in primary_details['weekly_plan'] if d['intensity'] == 'Rest')} days</p>
                <p><strong>High Intensity Days:</strong> {sum(1 for d in primary_details['weekly_plan'] if d['intensity'] == 'High')} days</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Alternative Workouts
    st.markdown("### 🔄 Alternative Workout Plans")
    st.markdown("*Try these if you want variety or if the primary plan doesn't fit your schedule*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    for idx, alt in enumerate(workout['alternatives'], 1):
        with st.expander(f"**Alternative {idx}: {alt['name']}**", expanded=False):
            alt_details = alt['details']
            
            if alt_details and 'weekly_plan' in alt_details:
                st.markdown(f"**Frequency:** {alt_details.get('frequency', '4-5 days/week')}")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Show simplified weekly view
                for day_plan in alt_details['weekly_plan']:
                    st.markdown(f"""
                        <div class="meal-card">
                            <strong>{day_plan['day']}</strong>: {day_plan['type']} 
                            ({day_plan['duration']}, {day_plan['intensity']} intensity)
                        </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # General Tips
    display_workout_tips()

def display_workout_tips():
    """Display general workout tips"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h4>💡 Before Your Workout</h4>
                <ul style="line-height: 1.8;">
                    <li>Warm up for 5-10 minutes</li>
                    <li>Stay hydrated (drink water 30 min before)</li>
                    <li>Eat a light snack if needed</li>
                    <li>Wear appropriate clothing</li>
                    <li>Check your equipment</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="success-box">
                <h4>✅ During Your Workout</h4>
                <ul style="line-height: 1.8;">
                    <li>Focus on proper form</li>
                    <li>Breathe properly</li>
                    <li>Listen to your body</li>
                    <li>Take breaks when needed</li>
                    <li>Stay motivated!</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Important Safety Notes</h4>
            <ul style="line-height: 1.8;">
                <li>Always consult a doctor before starting a new exercise program</li>
                <li>Stop immediately if you feel dizzy, nauseous, or experience pain</li>
                <li>Rest adequately between workout days</li>
                <li>Progress gradually - don't push too hard too fast</li>
                <li>Consider working with a certified trainer for complex exercises</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    """
Meal Plans Display Section
"""

def display_meal_plans(rec):
    """Display 7-day meal plan"""
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    meal_plan = rec['meal_plan']
    
    # Summary
    st.markdown(f"""
        <div class="info-box">
            <h2 style="margin: 0;">🍽️ Your 7-Day Meal Plan</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Goal: <strong>{rec['goal']}</strong> | 
                Target: <strong>{int(rec['target_calories'])} calories/day</strong>
            </p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.9;">
                💡 This plan repeats every week for the duration of your goal ({rec['target_days']} days)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Macro breakdown
    display_macro_breakdown(meal_plan)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Day selector
    st.markdown("### 📅 Daily Meal Plans")
    
    selected_day = st.selectbox(
        "Select a day to view details",
        options=list(meal_plan.keys()),
        format_func=lambda x: f"{x} ({int(meal_plan[x]['total_calories'])} cal)"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display selected day
    day_plan = meal_plan[selected_day]
    
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0 0 1rem 0;">{selected_day}</h3>
            <div style="display: flex; justify-content: space-around; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Target Calories</p>
                    <h3 style="margin: 0.25rem 0; color: #667eea;">{int(day_plan['target_calories'])}</h3>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Actual Calories</p>
                    <h3 style="margin: 0.25rem 0; color: #4caf50;">{int(day_plan['total_calories'])}</h3>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Difference</p>
                    <h3 style="margin: 0.25rem 0; color: {'#4caf50' if abs(day_plan['total_calories'] - day_plan['target_calories']) < 100 else '#ff9800'};">
                        {int(day_plan['total_calories'] - day_plan['target_calories']):+d}
                    </h3>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display meals
    for meal in day_plan['meals']:
        st.markdown(f"""
            <div class="meal-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div>
                        <h4 style="margin: 0; color: #667eea;">{meal['meal_type']}</h4>
                        <h3 style="margin: 0.25rem 0 0 0; font-size: 1.3rem;">{meal['dish_name']}</h3>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.5rem; font-weight: 700; color: #4caf50;">{meal['calories']}</span>
                        <span style="font-size: 0.9rem; opacity: 0.8;"> kcal</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; padding-top: 0.75rem; border-top: 1px solid #e0e0e0;">
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 0.85rem; opacity: 0.7;">Protein</p>
                        <p style="margin: 0.25rem 0 0 0; font-weight: 600;">{meal['protein']}g</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 0.85rem; opacity: 0.7;">Carbs</p>
                        <p style="margin: 0.25rem 0 0 0; font-weight: 600;">{meal['carbs']}g</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 0.85rem; opacity: 0.7;">Fats</p>
                        <p style="margin: 0.25rem 0 0 0; font-weight: 600;">{meal['fats']}g</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 0.85rem; opacity: 0.7;">Fiber</p>
                        <p style="margin: 0.25rem 0 0 0; font-weight: 600;">{meal['fiber']}g</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Quick view of all days
    with st.expander("📋 Quick View: All 7 Days", expanded=False):
        for day_name, day_data in meal_plan.items():
            st.markdown(f"**{day_name}** ({int(day_data['total_calories'])} cal)")
            for meal in day_data['meals']:
                st.markdown(f"- *{meal['meal_type']}*: {meal['dish_name']} ({meal['calories']} cal)")
            st.markdown("---")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Nutrition tips
    display_nutrition_tips(rec['goal'])

def display_macro_breakdown(meal_plan):
    """Display macronutrient breakdown chart"""
    
    # Calculate average macros
    total_protein = 0
    total_carbs = 0
    total_fats = 0
    num_days = len(meal_plan)
    
    for day_data in meal_plan.values():
        for meal in day_data['meals']:
            total_protein += meal['protein']
            total_carbs += meal['carbs']
            total_fats += meal['fats']
    
    avg_protein = total_protein / num_days
    avg_carbs = total_carbs / num_days
    avg_fats = total_fats / num_days
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Protein', 'Carbohydrates', 'Fats'],
        values=[avg_protein, avg_carbs, avg_fats],
        marker=dict(colors=['#667eea', '#764ba2', '#f093fb']),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title="Average Daily Macronutrient Distribution",
        height=350,
        showlegend=True,
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e3c72')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_nutrition_tips(goal):
    """Display goal-specific nutrition tips"""
    
    tips = {
        'Weight Loss': {
            'title': '🔥 Weight Loss Tips',
            'tips': [
                'Stay in a calorie deficit consistently',
                'Eat protein with every meal to preserve muscle',
                'Increase fiber intake for better satiety',
                'Avoid liquid calories (sugary drinks)',
                'Eat slowly and mindfully',
                'Stay hydrated - often thirst feels like hunger'
            ]
        },
        'Weight Gain': {
            'title': '💪 Weight Gain Tips',
            'tips': [
                'Eat more frequently (5-6 meals per day)',
                'Include calorie-dense foods (nuts, oils, avocados)',
                'Drink calories (smoothies, milk, juice)',
                'Add healthy fats to meals',
                'Don\'t skip meals',
                'Track your intake to ensure surplus'
            ]
        },
        'Muscle Building': {
            'title': '🏋️ Muscle Building Tips',
            'tips': [
                'Consume 1.6-2.2g protein per kg body weight',
                'Time protein intake around workouts',
                'Eat complex carbs for energy',
                'Stay in a slight calorie surplus',
                'Don\'t neglect healthy fats',
                'Get adequate sleep for recovery'
            ]
        },
        'Maintenance': {
            'title': '⚖️ Maintenance Tips',
            'tips': [
                'Balance your macronutrients',
                'Eat a variety of foods',
                'Listen to hunger cues',
                'Stay consistent',
                'Allow flexibility for social events',
                'Focus on whole foods'
            ]
        }
    }
    
    goal_tips = tips.get(goal, tips['Maintenance'])
    
    st.markdown(f"""
        <div class="success-box">
            <h4>{goal_tips['title']}</h4>
            <ul style="line-height: 2;">
                {''.join(f'<li>{tip}</li>' for tip in goal_tips['tips'])}
            </ul>
        </div>
    """, unsafe_allow_html=True)
    """
Progress Tracking Display Section
"""

def display_progress_tracking(rec):
    """Display progress tracking and projections"""
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    weight_proj = rec['weight_projection']
    
    # Weight projection summary
    st.markdown(f"""
        <div class="info-box">
            <h2 style="margin: 0;">📊 Your Weight Journey Projection</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem;">
                Based on your goal and calorie plan
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Current Weight</h4>
                <div class="stat-number" style="font-size: 2rem;">{weight_proj['current_weight']}</div>
                <p style="margin: 0; font-size: 0.85rem;">kg</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Target Weight</h4>
                <div class="stat-number" style="font-size: 2rem; color: #4caf50 !important;">{weight_proj['target_weight']}</div>
                <p style="margin: 0; font-size: 0.85rem;">kg</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        change_color = '#4caf50' if weight_proj['total_change'] < 0 else '#ff9800'
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Total Change</h4>
                <div class="stat-number" style="font-size: 2rem; color: {change_color} !important;">
                    {weight_proj['total_change']:+.1f}
                </div>
                <p style="margin: 0; font-size: 0.85rem;">kg</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Per Week</h4>
                <div class="stat-number" style="font-size: 2rem;">{abs(weight_proj['weekly_change']):.2f}</div>
                <p style="margin: 0; font-size: 0.85rem;">kg/week</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Weight projection chart
    display_weight_projection_chart(weight_proj, rec['target_days'])
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Weekly milestones
    display_weekly_milestones(weight_proj, rec['target_days'])
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Calorie breakdown
    display_calorie_breakdown(rec)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # BMI progression
    display_bmi_progression(weight_proj, st.session_state.user_profile['height_cm'])

def display_weight_projection_chart(weight_proj, target_days):
    """Display weight progression chart"""
    
    st.markdown("### 📈 Weight Progression Chart")
    
    # Generate data points
    current_weight = weight_proj['current_weight']
    target_weight = weight_proj['target_weight']
    total_change = weight_proj['total_change']
    
    # Create weekly data points
    weeks = target_days // 7
    if weeks == 0:
        weeks = 1
    
    weeks_array = list(range(0, weeks + 1))
    weights = [current_weight + (total_change * (week / weeks)) for week in weeks_array]
    
    # Create figure
    fig = go.Figure()
    
    # Projected weight line
    fig.add_trace(go.Scatter(
        x=weeks_array,
        y=weights,
        mode='lines+markers',
        name='Projected Weight',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#667eea'),
        hovertemplate='Week %{x}<br>Weight: %{y:.1f} kg<extra></extra>'
    ))
    
    # Target line
    fig.add_trace(go.Scatter(
        x=[0, weeks],
        y=[target_weight, target_weight],
        mode='lines',
        name='Target Weight',
        line=dict(color='#4caf50', width=2, dash='dash'),
        hovertemplate='Target: %{y:.1f} kg<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Weight Projection Over {weeks} Weeks",
        xaxis_title="Week",
        yaxis_title="Weight (kg)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        font=dict(color='#1e3c72', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

def display_weekly_milestones(weight_proj, target_days):
    """Display weekly weight milestones"""
    
    st.markdown("### 🎯 Weekly Milestones")
    
    current_weight = weight_proj['current_weight']
    total_change = weight_proj['total_change']
    weeks = max(1, target_days // 7)
    
    # Show up to 8 weeks
    display_weeks = min(weeks, 8)
    
    for week in range(1, display_weeks + 1):
        expected_weight = current_weight + (total_change * (week / weeks))
        change_so_far = expected_weight - current_weight
        
        progress = (week / weeks) * 100
        
        st.markdown(f"""
            <div class="meal-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0;">Week {week}</h4>
                        <p style="margin: 0.25rem 0 0 0;">Expected Weight: <strong>{expected_weight:.1f} kg</strong></p>
                        <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
                            Change from start: {change_so_far:+.1f} kg
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <div style="width: 60px; height: 60px; border-radius: 50%; 
                             background: conic-gradient(#667eea 0% {progress}%, #e0e0e0 {progress}% 100%);
                             display: flex; align-items: center; justify-content: center;">
                            <div style="width: 50px; height: 50px; border-radius: 50%; background: white;
                                 display: flex; align-items: center; justify-content: center;">
                                <span style="font-weight: 700; color: #667eea;">{int(progress)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if weeks > 8:
        st.info(f"💡 Showing first 8 weeks. Your full plan is {weeks} weeks ({target_days} days)")

def display_calorie_breakdown(rec):
    """Display calorie distribution chart"""
    
    st.markdown("### 🔥 Daily Calorie Breakdown")
    
    maintenance = rec['maintenance_calories']
    target = rec['target_calories']
    bmr = rec['bmr']
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='BMR (Resting)',
        x=['Calories'],
        y=[bmr],
        marker_color='#667eea',
        text=[f'{int(bmr)} kcal'],
        textposition='inside',
        hovertemplate='BMR: %{y:.0f} kcal<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Activity Burn',
        x=['Calories'],
        y=[maintenance - bmr],
        marker_color='#764ba2',
        text=[f'{int(maintenance - bmr)} kcal'],
        textposition='inside',
        hovertemplate='Activity: %{y:.0f} kcal<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Goal Adjustment',
        x=['Calories'],
        y=[target - maintenance],
        marker_color='#f093fb' if (target - maintenance) > 0 else '#4caf50',
        text=[f'{int(target - maintenance):+d} kcal'],
        textposition='inside',
        hovertemplate='Adjustment: %{y:+.0f} kcal<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='relative',
        height=350,
        showlegend=True,
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        font=dict(color='#1e3c72'),
        xaxis=dict(showticklabels=False),
        yaxis_title="Calories (kcal)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maintenance", f"{int(maintenance)} kcal", "What you burn")
    with col2:
        st.metric("Target", f"{int(target)} kcal", "What you eat")
    with col3:
        diff = target - maintenance
        st.metric("Daily Deficit/Surplus", f"{int(diff):+d} kcal", 
                 "Deficit" if diff < 0 else "Surplus")

def display_bmi_progression(weight_proj, height_cm):
    """Display BMI progression"""
    
    st.markdown("### 📏 BMI Progression")
    
    current_bmi = weight_proj['current_weight'] / ((height_cm/100) ** 2)
    target_bmi = weight_proj['target_weight'] / ((height_cm/100) ** 2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0;">Current BMI</h4>
                <div class="stat-number" style="font-size: 2.5rem;">{current_bmi:.1f}</div>
                <p style="margin: 0; color: {get_bmi_color(current_bmi)} !important; font-weight: 600;">
                    {get_bmi_status(current_bmi)}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0;">Target BMI</h4>
                <div class="stat-number" style="font-size: 2.5rem; color: #4caf50 !important;">{target_bmi:.1f}</div>
                <p style="margin: 0; color: {get_bmi_color(target_bmi)} !important; font-weight: 600;">
                    {get_bmi_status(target_bmi)}
                </p>
            </div>
        """, unsafe_allow_html=True)

def get_bmi_color(bmi):
    """Get color based on BMI status"""
    if bmi < 18.5 or bmi >= 30:
        return '#f44336'
    elif 18.5 <= bmi < 25:
        return '#4caf50'
    else:
        return '#ff9800'
"""
Hydration Plan and Progress Logging
"""

def display_hydration_plan(rec):
    """Display hydration plan and reminders"""
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    hydration = rec['hydration']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
            <div class="info-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
                <h2 style="margin: 0; color: white !important;">💧 Your Daily Hydration Goal</h2>
                <div style="text-align: center; padding: 2rem 0;">
                    <div style="font-size: 4rem; font-weight: 700;">{hydration['daily_water_liters']}</div>
                    <div style="font-size: 1.5rem;">Liters / Day</div>
                    <div style="font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem;">
                        ({hydration['daily_water_ml']} ml)
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0;">📱 Reminder Schedule</h4>
                <p><strong>Frequency:</strong> {hydration['num_reminders']} times per day</p>
                <p><strong>Amount per reminder:</strong> {hydration['ml_per_reminder']} ml</p>
                <p><strong>Interval:</strong> Every {hydration['interval_minutes']} minutes</p>
                <div class="success-box" style="margin-top: 1rem;">
                    <p style="margin: 0;">💡 <strong>Pro Tip:</strong> Drink a full glass (250ml) immediately after waking up!</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0;">⏰ Suggested Timings</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate timing suggestions
        start_hour = 7
        for i in range(hydration['num_reminders']):
            hour = start_hour + (i * (hydration['interval_minutes'] / 60))
            time_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"
            
            if i < 3:
                color, label = "#fff3cd", "🌅 Morning"
            elif i < 6:
                color, label = "#d1ecf1", "☀️ Afternoon"
            else:
                color, label = "#e2d5f7", "🌙 Evening"
            
            st.markdown(f"""
                <div style='background:{color}; padding:12px; border-radius:10px; 
                     margin:8px 0; display:flex; justify-content:space-between; align-items: center;'>
                    <span style='font-weight:600; color: #1e3c72;'>{time_str}</span>
                    <span style='color: #1e3c72;'>{label} - {hydration['ml_per_reminder']}ml</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Benefits
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="success-box">
                <h4>✨ Hydration Benefits</h4>
                <ul style="line-height: 2;">
                    <li>💪 Better workout performance</li>
                    <li>🧠 Improved mental clarity</li>
                    <li>⚡ More energy throughout day</li>
                    <li>🌡️ Better temperature regulation</li>
                    <li>✨ Healthier skin</li>
                    <li>🏃 Faster recovery</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box">
                <h4>💡 Hydration Tips</h4>
                <ul style="line-height: 2;">
                    <li>Carry a water bottle everywhere</li>
                    <li>Set phone reminders</li>
                    <li>Drink before you feel thirsty</li>
                    <li>Add lemon/cucumber for flavor</li>
                    <li>Increase intake during workouts</li>
                    <li>Monitor urine color (light yellow is good)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def display_progress_logging():
    """Display progress logging interface"""
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h2 style="margin: 0;">📝 Log Your Daily Progress</h2>
            <p style="margin: 0.5rem 0 0 0;">Track your journey and stay accountable!</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Logging form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Today's Log")
        
        with st.form("progress_log_form"):
            log_date = st.date_input("Date", datetime.now())
            
            current_weight = st.number_input(
                "Current Weight (kg)",
                min_value=30.0,
                max_value=200.0,
                value=float(st.session_state.user_profile.get('weight_kg', 70)),
                step=0.1
            )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                workout_completed = st.checkbox("✅ Completed today's workout")
                calories_consumed = st.number_input("Calories Consumed", 0, 5000, 2000, 50)
            
            with col_b:
                water_intake = st.number_input("Water Intake (L)", 0.0, 10.0, 2.0, 0.1)
                sleep_hours = st.number_input("Hours of Sleep", 0.0, 12.0, 7.0, 0.5)
            
            notes = st.text_area("Notes / How are you feeling?", 
                                placeholder="Any observations, challenges, or victories to note...")
            
            submit_log = st.form_submit_button("💾 Save Progress", use_container_width=True)
            
            if submit_log:
                progress_data = {
                    'date': str(log_date),
                    'weight': current_weight,
                    'workout_completed': workout_completed,
                    'calories_consumed': calories_consumed,
                    'water_intake': water_intake,
                    'sleep_hours': sleep_hours,
                    'notes': notes
                }
                
                result = st.session_state.auth_system.log_progress(
                    st.session_state.username,
                    progress_data
                )
                
                if result['success']:
                    st.success("✅ Progress logged successfully!")
                    st.balloons()
                    
                    # Update user weight in profile
                    profile_update = {'weight_kg': current_weight}
                    st.session_state.auth_system.update_user_profile(
                        st.session_state.username,
                        profile_update
                    )
                    st.session_state.user_profile['weight_kg'] = current_weight
                else:
                    st.error(f"❌ {result['message']}")
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        # Get progress history
        history = st.session_state.auth_system.get_progress_history(
            st.session_state.username,
            days=7
        )
        
        if history:
            recent_log = history[-1]
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0;">Last Logged</h4>
                    <p style="margin: 0.25rem 0;"><strong>Date:</strong> {recent_log.get('date', 'N/A')}</p>
                    <p style="margin: 0.25rem 0;"><strong>Weight:</strong> {recent_log.get('weight', 'N/A')} kg</p>
                    <p style="margin: 0.25rem 0;"><strong>Calories:</strong> {recent_log.get('calories_consumed', 'N/A')} kcal</p>
                    <p style="margin: 0.25rem 0;"><strong>Water:</strong> {recent_log.get('water_intake', 'N/A')} L</p>
                    <p style="margin: 0.25rem 0;"><strong>Workout:</strong> {'✅ Yes' if recent_log.get('workout_completed') else '❌ No'}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show streak
            streak = sum(1 for log in history[-7:] if log.get('workout_completed', False))
            st.markdown(f"""
                <div class="success-box" style="text-align: center;">
                    <h3 style="margin: 0;">🔥 {streak} Day Streak</h3>
                    <p style="margin: 0.5rem 0 0 0;">Keep it up!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No progress logged yet. Start tracking today!")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Progress history
    if history and len(history) > 0:
        st.markdown("### 📈 Progress History (Last 7 Days)")
        
        # Convert to DataFrame
        df_history = pd.DataFrame(history[-7:])
        
        if not df_history.empty and 'weight' in df_history.columns:
            # Weight progress chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_history['date'],
                y=df_history['weight'],
                mode='lines+markers',
                name='Weight',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Weight Progress",
                xaxis_title="Date",
                yaxis_title="Weight (kg)",
                height=350,
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(255,255,255,0.9)',
                font=dict(color='#1e3c72')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.markdown("#### Recent Logs")
            display_df = df_history[['date', 'weight', 'calories_consumed', 'water_intake', 'workout_completed']].tail(7)
            display_df.columns = ['Date', 'Weight (kg)', 'Calories', 'Water (L)', 'Workout Done']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            """
Admin Panel and Main App Assembly
"""

def display_admin_panel():
    """Admin panel for user management"""
    
    # Check if user is admin (you can set admin username here)
    if st.session_state.username not in ['admin', 'shivashankar']:  # Add your admin username
        st.error("❌ Access Denied: Admin privileges required")
        return
    
    st.markdown("""
        <div class="info-box" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white;">
            <h1 style="margin: 0; color: white !important;">🔧 Admin Dashboard</h1>
            <p style="margin: 0.5rem 0 0 0; color: white !important;">Manage users and view statistics</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get all users
    users = st.session_state.auth_system.get_all_users()
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0;">Total Users</h4>
                <div class="stat-number" style="font-size: 2.5rem;">{len(users)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_users = sum(1 for u in users if u.get('last_login'))
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0;">Active Users</h4>
                <div class="stat-number" style="font-size: 2.5rem;">{active_users}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        complete_profiles = sum(1 for u in users if u.get('profile_complete'))
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0;">Complete Profiles</h4>
                <div class="stat-number" style="font-size: 2.5rem;">{complete_profiles}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        new_today = sum(1 for u in users if u.get('created_at', '').startswith(str(datetime.now().date())))
        st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0;">New Today</h4>
                <div class="stat-number" style="font-size: 2.5rem;">{new_today}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # User table
    st.markdown("### 👥 User Management")
    
    if users:
        df_users = pd.DataFrame(users)
        
        # Format the dataframe
        display_df = df_users[['username', 'email', 'created_at', 'last_login', 'profile_complete']].copy()
        display_df.columns = ['Username', 'Email', 'Created At', 'Last Login', 'Profile Complete']
        
        # Format dates
        for col in ['Created At', 'Last Login']:
            if col in display_df.columns:
                display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                display_df[col] = display_df[col].fillna('Never')
        
        # Display dataframe
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Delete user section
        with st.expander("🗑️ Delete User", expanded=False):
            st.warning("⚠️ This action cannot be undone!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                user_to_delete = st.selectbox(
                    "Select user to delete",
                    options=[u['username'] for u in users if u['username'] not in ['admin', 'shivashankar']]
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Delete User", type="primary"):
                    if user_to_delete:
                        result = st.session_state.auth_system.delete_user(user_to_delete)
                        if result['success']:
                            st.success(f"✅ User {user_to_delete} deleted successfully")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
    else:
        st.info("No users found")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Back to dashboard button
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.rerun()

# Main Application Router
def main():
    """Main application logic"""
    
    init_session_state()
    
    if not st.session_state.logged_in:
        # Show login page
        login_page()
    else:
        # Check if admin panel requested
        if st.session_state.get('show_admin', False):
            display_admin_panel()
        else:
            # Show main dashboard with sidebar
            profile_setup_sidebar()
            display_dashboard()
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem; opacity: 0.7;'>
            <p style='font-size: 0.9rem;'>Built with ❤️ using Machine Learning, Scikit-learn & Streamlit</p>
            <p style='font-size: 0.85rem; margin-top: 0.5rem;'>
                ⚠️ <strong>Disclaimer:</strong> Always consult healthcare professionals before starting any fitness program
            </p>
            <p style='font-size: 0.8rem; margin-top: 0.5rem;'>© 2024 FitLife Pro. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
