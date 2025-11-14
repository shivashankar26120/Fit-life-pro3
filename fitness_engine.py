"""
Enhanced Fitness Recommendation Engine
Handles all ML models, calculations, and business logic
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FitnessEngine:
    """Main engine for fitness recommendations"""

    def __init__(self):
        self.knn_model = None
        self.rf_model = None
        self.lr_model = None
        self.scaler_workout = None
        self.scaler_cal = None
        self.le_workout = None
        self.le_gender = None
        self.df_food = None
        self.workout_library = self._load_workout_library()

    def _load_workout_library(self):
        """Comprehensive workout library with 7 different plans"""
        return {
            'Running': {
                'primary': {
                    'name': 'Progressive Running Program',
                    'frequency': '5 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Easy Run', 'duration': '30 min', 'intensity': 'Low'},
                        {'day': 'Tuesday', 'type': 'Interval Training', 'duration': '25 min', 'intensity': 'High'},
                        {'day': 'Wednesday', 'type': 'Rest/Cross-training', 'duration': '30 min', 'intensity': 'Low'},
                        {'day': 'Thursday', 'type': 'Tempo Run', 'duration': '35 min', 'intensity': 'Medium'},
                        {'day': 'Friday', 'type': 'Rest', 'duration': '0 min', 'intensity': 'Rest'},
                        {'day': 'Saturday', 'type': 'Long Run', 'duration': '45-60 min', 'intensity': 'Medium'},
                        {'day': 'Sunday', 'type': 'Recovery Run', 'duration': '20 min', 'intensity': 'Low'}
                    ]
                },
                'alternatives': ['Swimming', 'Cycling']
            },
            'Swimming': {
                'primary': {
                    'name': 'Aquatic Excellence Program',
                    'frequency': '4-5 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Technique Focus', 'duration': '45 min', 'intensity': 'Medium'},
                        {'day': 'Tuesday', 'type': 'Endurance Swim', 'duration': '60 min', 'intensity': 'Low'},
                        {'day': 'Wednesday', 'type': 'Sprint Intervals', 'duration': '40 min', 'intensity': 'High'},
                        {'day': 'Thursday', 'type': 'Active Recovery', 'duration': '30 min', 'intensity': 'Low'},
                        {'day': 'Friday', 'type': 'Mixed Strokes', 'duration': '50 min', 'intensity': 'Medium'},
                        {'day': 'Saturday', 'type': 'Long Distance', 'duration': '70 min', 'intensity': 'Medium'},
                        {'day': 'Sunday', 'type': 'Rest', 'duration': '0 min', 'intensity': 'Rest'}
                    ]
                },
                'alternatives': ['Cycling', 'Yoga']
            },
            'Cycling': {
                'primary': {
                    'name': 'Power Cycling Program',
                    'frequency': '4-5 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Flat Route Easy', 'duration': '45 min', 'intensity': 'Low'},
                        {'day': 'Tuesday', 'type': 'Hill Repeats', 'duration': '50 min', 'intensity': 'High'},
                        {'day': 'Wednesday', 'type': 'Rest/Stretching', 'duration': '20 min', 'intensity': 'Low'},
                        {'day': 'Thursday', 'type': 'Tempo Ride', 'duration': '60 min', 'intensity': 'Medium'},
                        {'day': 'Friday', 'type': 'Recovery Spin', 'duration': '30 min', 'intensity': 'Low'},
                        {'day': 'Saturday', 'type': 'Long Endurance', 'duration': '90-120 min', 'intensity': 'Medium'},
                        {'day': 'Sunday', 'type': 'Rest', 'duration': '0 min', 'intensity': 'Rest'}
                    ]
                },
                'alternatives': ['Running', 'Swimming']
            },
            'Weight Training': {
                'primary': {
                    'name': 'Strength Building Program',
                    'frequency': '4-5 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Upper Body Push', 'duration': '60 min', 'intensity': 'High',
                         'exercises': 'Bench Press, Shoulder Press, Tricep Dips'},
                        {'day': 'Tuesday', 'type': 'Lower Body', 'duration': '60 min', 'intensity': 'High',
                         'exercises': 'Squats, Deadlifts, Lunges'},
                        {'day': 'Wednesday', 'type': 'Active Recovery', 'duration': '30 min', 'intensity': 'Low',
                         'exercises': 'Light Cardio, Stretching'},
                        {'day': 'Thursday', 'type': 'Upper Body Pull', 'duration': '60 min', 'intensity': 'High',
                         'exercises': 'Pull-ups, Rows, Bicep Curls'},
                        {'day': 'Friday', 'type': 'Full Body', 'duration': '50 min', 'intensity': 'Medium',
                         'exercises': 'Compound Movements'},
                        {'day': 'Saturday', 'type': 'Core & Conditioning', 'duration': '40 min', 'intensity': 'Medium',
                         'exercises': 'Planks, Cable Work, Abs'},
                        {'day': 'Sunday', 'type': 'Rest', 'duration': '0 min', 'intensity': 'Rest'}
                    ]
                },
                'alternatives': ['HIIT', 'Yoga']
            },
            'HIIT': {
                'primary': {
                    'name': 'High-Intensity Interval Training',
                    'frequency': '3-4 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Full Body HIIT', 'duration': '30 min', 'intensity': 'High',
                         'exercises': 'Burpees, Jump Squats, Mountain Climbers'},
                        {'day': 'Tuesday', 'type': 'Active Recovery', 'duration': '30 min', 'intensity': 'Low',
                         'exercises': 'Walking, Light Yoga'},
                        {'day': 'Wednesday', 'type': 'Cardio HIIT', 'duration': '25 min', 'intensity': 'High',
                         'exercises': 'Sprint Intervals, Jump Rope'},
                        {'day': 'Thursday', 'type': 'Rest', 'duration': '0 min', 'intensity': 'Rest'},
                        {'day': 'Friday', 'type': 'Strength HIIT', 'duration': '35 min', 'intensity': 'High',
                         'exercises': 'Kettlebell Swings, Box Jumps, Battle Ropes'},
                        {'day': 'Saturday', 'type': 'Moderate Cardio', 'duration': '40 min', 'intensity': 'Medium',
                         'exercises': 'Jogging, Cycling'},
                        {'day': 'Sunday', 'type': 'Rest/Stretching', 'duration': '20 min', 'intensity': 'Low'}
                    ]
                },
                'alternatives': ['Weight Training', 'Running']
            },
            'Yoga': {
                'primary': {
                    'name': 'Complete Yoga Program',
                    'frequency': '5-6 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Vinyasa Flow', 'duration': '60 min', 'intensity': 'Medium'},
                        {'day': 'Tuesday', 'type': 'Power Yoga', 'duration': '50 min', 'intensity': 'High'},
                        {'day': 'Wednesday', 'type': 'Hatha Yoga', 'duration': '45 min', 'intensity': 'Low'},
                        {'day': 'Thursday', 'type': 'Yin Yoga', 'duration': '60 min', 'intensity': 'Low'},
                        {'day': 'Friday', 'type': 'Ashtanga', 'duration': '75 min', 'intensity': 'Medium'},
                        {'day': 'Saturday', 'type': 'Restorative', 'duration': '60 min', 'intensity': 'Low'},
                        {'day': 'Sunday', 'type': 'Meditation & Pranayama', 'duration': '30 min', 'intensity': 'Low'}
                    ]
                },
                'alternatives': ['Dancing', 'Swimming']
            },
            'Dancing': {
                'primary': {
                    'name': 'Dance Fitness Program',
                    'frequency': '4-5 days/week',
                    'weekly_plan': [
                        {'day': 'Monday', 'type': 'Zumba', 'duration': '45 min', 'intensity': 'Medium'},
                        {'day': 'Tuesday', 'type': 'Hip Hop', 'duration': '50 min', 'intensity': 'High'},
                        {'day': 'Wednesday', 'type': 'Stretching', 'duration': '30 min', 'intensity': 'Low'},
                        {'day': 'Thursday', 'type': 'Bollywood Dance', 'duration': '60 min', 'intensity': 'Medium'},
                        {'day': 'Friday', 'type': 'Contemporary', 'duration': '45 min', 'intensity': 'Medium'},
                        {'day': 'Saturday', 'type': 'Dance Cardio', 'duration': '55 min', 'intensity': 'High'},
                        {'day': 'Sunday', 'type': 'Rest/Light Movement', 'duration': '20 min', 'intensity': 'Low'}
                    ]
                },
                'alternatives': ['Yoga', 'HIIT']
            }
        }

    def load_and_train_models(self, health_csv, food_csv):
        """Load datasets and train all models with improved accuracy"""
        try:
            # Load data
            df_health = pd.read_csv('/content/drive/MyDrive/health_fitness_dataset.csv')
            self.df_food = pd.read_csv('/content/drive/MyDrive/Indian_Food_Nutrition_Processed.csv')

            # Create user profiles
            user_profiles = df_health.groupby('participant_id').agg({
                'age': 'first',
                'gender': 'first',
                'height_cm': 'first',
                'weight_kg': 'mean',
                'bmi': 'mean',
                'fitness_level': 'mean',
                'daily_steps': 'mean',
                'activity_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
                'calories_burned': 'mean',
                'avg_heart_rate': 'mean',
                'stress_level': 'mean'
            }).reset_index()

            # Calculate BMR
            user_profiles['bmr'] = user_profiles.apply(self._calculate_bmr, axis=1)
            user_profiles['activity_mult'] = user_profiles['daily_steps'].apply(self._get_activity_multiplier)
            user_profiles['maintenance_calories'] = user_profiles['bmr'] * user_profiles['activity_mult']

            # Train workout model (Using Random Forest for better accuracy)
            X_workout = user_profiles[['age', 'bmi', 'fitness_level', 'daily_steps', 'stress_level']]
            y_workout = user_profiles['activity_type']

            self.le_workout = LabelEncoder()
            y_encoded = self.le_workout.fit_transform(y_workout)

            self.scaler_workout = StandardScaler()
            X_scaled = self.scaler_workout.fit_transform(X_workout)

            # Use Random Forest for better accuracy
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            self.rf_model.fit(X_scaled, y_encoded)

            # Also train KNN as backup
            self.knn_model = KNeighborsClassifier(n_neighbors=7)
            self.knn_model.fit(X_scaled, y_encoded)

            # Train calorie model
            X_cal = user_profiles[['age', 'height_cm', 'weight_kg', 'bmi', 'daily_steps', 'fitness_level']]
            self.le_gender = LabelEncoder()
            X_cal['gender'] = self.le_gender.fit_transform(user_profiles['gender'])

            y_cal = user_profiles['maintenance_calories']

            self.scaler_cal = StandardScaler()
            X_cal_scaled = self.scaler_cal.fit_transform(X_cal)

            self.lr_model = LinearRegression()
            self.lr_model.fit(X_cal_scaled, y_cal)

            # Clean food data
            self.df_food['Vitamin C (mg)'].fillna(self.df_food['Vitamin C (mg)'].median(), inplace=True)
            self.df_food['Folate (µg)'].fillna(self.df_food['Folate (µg)'].median(), inplace=True)

            # Calculate model accuracy
            cv_scores = cross_val_score(self.rf_model, X_scaled, y_encoded, cv=5)
            accuracy = cv_scores.mean()

            return {
                'success': True,
                'accuracy': accuracy,
                'num_users': len(user_profiles),
                'num_foods': len(self.df_food)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_bmr(self, row):
        """Calculate Basal Metabolic Rate"""
        if row['gender'] == 'M':
            return 10 * row['weight_kg'] + 6.25 * row['height_cm'] - 5 * row['age'] + 5
        else:
            return 10 * row['weight_kg'] + 6.25 * row['height_cm'] - 5 * row['age'] - 161

    def _get_activity_multiplier(self, steps):
        """Get activity multiplier based on steps"""
        if steps < 3000:
            return 1.2
        elif steps < 7500:
            return 1.375
        elif steps < 10000:
            return 1.55
        else:
            return 1.725

    def predict_workout(self, user_data):
        """Predict optimal workout with alternatives"""
        X_new = np.array([[
            user_data['age'],
            user_data['bmi'],
            user_data['fitness_level'],
            user_data['daily_steps'],
            user_data.get('stress_level', 5)
        ]])

        X_scaled = self.scaler_workout.transform(X_new)

        # Get prediction from Random Forest
        prediction = self.rf_model.predict(X_scaled)[0]
        probabilities = self.rf_model.predict_proba(X_scaled)[0]

        # Get top 3 workouts
        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        primary_workout = self.le_workout.inverse_transform([prediction])[0]
        alternatives = [self.le_workout.classes_[i] for i in top_3_indices[1:]]

        # Get detailed workout plans
        primary_plan = self.workout_library.get(primary_workout, {}).get('primary', {})
        alternative_plans = [self.workout_library.get(alt, {}).get('primary', {}) for alt in alternatives]

        return {
            'primary': {
                'name': primary_workout,
                'details': primary_plan,
                'confidence': probabilities[prediction] * 100
            },
            'alternatives': [
                {'name': alt, 'details': alternative_plans[i]}
                for i, alt in enumerate(alternatives)
            ]
        }

    def calculate_target_weight(self, current_weight, goal, target_days, daily_calorie_deficit):
        """Calculate expected weight change"""
        # 1 kg = 7700 calories
        total_deficit = daily_calorie_deficit * target_days
        weight_change = total_deficit / 7700

        if goal in ['Weight Loss', 'weight_loss']:
            target_weight = current_weight - abs(weight_change)
            weekly_loss = abs(weight_change) / (target_days / 7)
        elif goal in ['Weight Gain', 'weight_gain']:
            target_weight = current_weight + abs(weight_change)
            weekly_loss = abs(weight_change) / (target_days / 7)
        else:
            target_weight = current_weight
            weekly_loss = 0

        return {
            'current_weight': round(current_weight, 1),
            'target_weight': round(target_weight, 1),
            'total_change': round(weight_change, 1),
            'weekly_change': round(weekly_loss, 2),
            'days': target_days
        }

    def recommend_smart_meals(self, goal, target_calories, num_days=7):
        """Enhanced meal recommendation with proper logic"""

        # Define macro ratios based on goal
        if goal in ['Weight Loss', 'weight_loss']:
            protein_ratio, carb_ratio, fat_ratio = 0.35, 0.40, 0.25
            calorie_adj = -500
        elif goal in ['Weight Gain', 'weight_gain']:
            protein_ratio, carb_ratio, fat_ratio = 0.30, 0.45, 0.25
            calorie_adj = 300
        elif goal in ['Muscle Building', 'muscle_building']:
            protein_ratio, carb_ratio, fat_ratio = 0.40, 0.40, 0.20
            calorie_adj = 200
        else:
            protein_ratio, carb_ratio, fat_ratio = 0.30, 0.45, 0.25
            calorie_adj = 0

        adjusted_calories = target_calories + calorie_adj
        meals_per_day = 5
        calories_per_meal = adjusted_calories / meals_per_day

        # Calculate target macros
        target_protein = (calories_per_meal * protein_ratio) / 4
        target_carbs = (calories_per_meal * carb_ratio) / 4
        target_fats = (calories_per_meal * fat_ratio) / 9

        # Score foods
        df_scored = self.df_food.copy()
        df_scored['score'] = 0

        for idx, row in df_scored.iterrows():
            # Calculate scores
            protein_score = 100 - min(abs(row['Protein (g)'] - target_protein) * 5, 100)
            carb_score = 100 - min(abs(row['Carbohydrates (g)'] - target_carbs) * 3, 100)
            fat_score = 100 - min(abs(row['Fats (g)'] - target_fats) * 5, 100)
            calorie_score = 100 - min(abs(row['Calories (kcal)'] - calories_per_meal) / 5, 100)

            # Bonuses
            fiber_bonus = row['Fibre (g)'] * 10
            protein_bonus = row['Protein (g)'] * 2  # Bonus for high protein

            # Penalties
            sodium_penalty = max(0, (row['Sodium (mg)'] - 500) / 10) if row['Sodium (mg)'] > 500 else 0
            sugar_penalty = row['Free Sugar (g)'] * 3

            total_score = (protein_score + carb_score + fat_score + calorie_score +
                          fiber_bonus + protein_bonus - sodium_penalty - sugar_penalty)

            df_scored.at[idx, 'score'] = max(0, total_score)

        # Get top dishes
        df_sorted = df_scored.sort_values('score', ascending=False)
        top_dishes = df_sorted.head(60).sample(frac=1, random_state=42).reset_index(drop=True)

        # Create meal plans
        meal_types = ['Breakfast', 'Mid-Morning Snack', 'Lunch', 'Evening Snack', 'Dinner']
        weekly_plan = {}

        for day in range(1, num_days + 1):
            daily_meals = []
            day_calories = 0
            start_idx = ((day - 1) * meals_per_day) % len(top_dishes)

            for meal_idx, meal_type in enumerate(meal_types):
                dish_idx = (start_idx + meal_idx) % len(top_dishes)
                dish = top_dishes.iloc[dish_idx]

                daily_meals.append({
                    'meal_type': meal_type,
                    'dish_name': dish['Dish Name'],
                    'calories': round(dish['Calories (kcal)'], 1),
                    'protein': round(dish['Protein (g)'], 1),
                    'carbs': round(dish['Carbohydrates (g)'], 1),
                    'fats': round(dish['Fats (g)'], 1),
                    'fiber': round(dish['Fibre (g)'], 1)
                })
                day_calories += dish['Calories (kcal)']

            weekly_plan[f'Day {day}'] = {
                'meals': daily_meals,
                'total_calories': round(day_calories, 0),
                'target_calories': round(adjusted_calories, 0),
                'calorie_adjustment': calorie_adj
            }

        return weekly_plan

    def calculate_hydration(self, weight_kg, activity_level, workout_intensity='Medium'):
        """Calculate daily water intake"""
        base_water = weight_kg * 35  # ml per kg

        activity_bonus = {
            'sedentary': 0,
            'lightly_active': 300,
            'moderately_active': 500,
            'very_active': 700
        }
        base_water += activity_bonus.get(activity_level, 0)

        if workout_intensity == 'High':
            base_water += 500
        elif workout_intensity == 'Medium':
            base_water += 300

        num_reminders = 10
        ml_per_reminder = base_water / num_reminders

        return {
            'daily_water_ml': round(base_water),
            'daily_water_liters': round(base_water/1000, 2),
            'ml_per_reminder': round(ml_per_reminder),
            'num_reminders': num_reminders,
            'interval_minutes': 90
        }
