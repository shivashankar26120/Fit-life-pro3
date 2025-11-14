"""
Secure User Authentication System
Handles user registration, login, and profile management
"""

import json
import os
import bcrypt
from datetime import datetime
from pathlib import Path


class AuthSystem:
    """Handles all authentication and user management"""

    def __init__(self, db_path='users_db.json'):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create database file if it doesn't exist"""
        if not os.path.exists(self.db_path):
            self._save_db({})

    def _load_db(self):
        """Load user database"""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_db(self, db):
        """Save user database"""
        with open(self.db_path, 'w') as f:
            json.dump(db, f, indent=2)

    def hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def register_user(self, username, password, email=''):
        """Register a new user"""
        db = self._load_db()

        if username in db:
            return {'success': False, 'message': 'Username already exists'}

        if len(password) < 6:
            return {'success': False, 'message': 'Password must be at least 6 characters'}

        # Create user
        db[username] = {
            'password_hash': self.hash_password(password),
            'email': email,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'profile': {
                'age': None,
                'gender': None,
                'height_cm': None,
                'weight_kg': None,
                'daily_steps': None,
                'fitness_level': None,
                'goal': None,
                'target_days': None,
                'target_weight_change': None
            },
            'workout_history': [],
            'meal_plans': [],
            'progress_logs': []
        }

        self._save_db(db)
        return {'success': True, 'message': 'Registration successful'}

    def login_user(self, username, password):
        """Authenticate user"""
        db = self._load_db()

        if username not in db:
            return {'success': False, 'message': 'Invalid username or password'}

        user = db[username]

        if not self.verify_password(password, user['password_hash']):
            return {'success': False, 'message': 'Invalid username or password'}

        # Update last login
        db[username]['last_login'] = datetime.now().isoformat()
        self._save_db(db)

        return {
            'success': True,
            'message': 'Login successful',
            'user_data': {
                'username': username,
                'profile': user.get('profile', {}),
                'last_login': user['last_login'],
                'created_at': user['created_at']
            }
        }

    def get_user_profile(self, username):
        """Get user profile data"""
        db = self._load_db()

        if username not in db:
            return None

        return db[username].get('profile', {})

    def update_user_profile(self, username, profile_data):
        """Update user profile"""
        db = self._load_db()

        if username not in db:
            return {'success': False, 'message': 'User not found'}

        # Update profile
        current_profile = db[username].get('profile', {})
        current_profile.update(profile_data)
        db[username]['profile'] = current_profile
        db[username]['updated_at'] = datetime.now().isoformat()

        self._save_db(db)
        return {'success': True, 'message': 'Profile updated'}

    def log_progress(self, username, progress_data):
        """Log user progress"""
        db = self._load_db()

        if username not in db:
            return {'success': False, 'message': 'User not found'}

        progress_entry = {
            'date': datetime.now().isoformat(),
            'weight': progress_data.get('weight'),
            'workout_completed': progress_data.get('workout_completed', False),
            'calories_consumed': progress_data.get('calories_consumed'),
            'water_intake': progress_data.get('water_intake'),
            'notes': progress_data.get('notes', '')
        }

        if 'progress_logs' not in db[username]:
            db[username]['progress_logs'] = []

        db[username]['progress_logs'].append(progress_entry)
        self._save_db(db)

        return {'success': True, 'message': 'Progress logged'}

    def get_progress_history(self, username, days=30):
        """Get user's progress history"""
        db = self._load_db()

        if username not in db:
            return []

        logs = db[username].get('progress_logs', [])
        return logs[-days:] if len(logs) > days else logs

    def save_meal_plan(self, username, meal_plan):
        """Save generated meal plan"""
        db = self._load_db()

        if username not in db:
            return {'success': False, 'message': 'User not found'}

        plan_entry = {
            'generated_at': datetime.now().isoformat(),
            'plan': meal_plan
        }

        if 'meal_plans' not in db[username]:
            db[username]['meal_plans'] = []

        db[username]['meal_plans'].append(plan_entry)

        # Keep only last 10 plans
        if len(db[username]['meal_plans']) > 10:
            db[username]['meal_plans'] = db[username]['meal_plans'][-10:]

        self._save_db(db)
        return {'success': True, 'message': 'Meal plan saved'}

    def get_all_users(self):
        """Get all users (for admin panel)"""
        db = self._load_db()
        users = []

        for username, data in db.items():
            users.append({
                'username': username,
                'email': data.get('email', ''),
                'created_at': data.get('created_at', ''),
                'last_login': data.get('last_login', ''),
                'profile_complete': all([
                    data.get('profile', {}).get('age'),
                    data.get('profile', {}).get('weight_kg'),
                    data.get('profile', {}).get('height_cm')
                ])
            })

        return users

    def delete_user(self, username):
        """Delete a user (admin function)"""
        db = self._load_db()

        if username in db:
            del db[username]
            self._save_db(db)
            return {'success': True, 'message': f'User {username} deleted'}

        return {'success': False, 'message': 'User not found'}
