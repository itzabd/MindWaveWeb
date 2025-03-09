import os

from sqlalchemy.engine import cursor
from supabase import create_client, Client
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set secret key from environment variables
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://okhrguykcmeaakxclkbl.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9raHJndXlrY21lYWFreGNsa2JsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAwNTM0NzMsImV4cCI6MjA1NTYyOTQ3M30.XytMV4yJ5GMmhD_53E4rAByqBSY8GcqD1C0B3IWBjh8")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials! Check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, id, name, email, password_hash, role="user"):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash
        self.role = role

    def __repr__(self):
        return f"<User {self.name}>"

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def get_by_email(email):
        try:
            # Execute the Supabase query
            response = supabase.table("users").select("*").eq("email", email).execute()

            # Check if the response was successful and contains data
            if response and response.data:
                user_data = response.data[0]
                # Filter out the 'password' field (if it exists)
                filtered_data = {
                    "id": user_data["id"],
                    "name": user_data["name"],
                    "email": user_data["email"],
                    "password_hash": user_data["password_hash"],
                    "role": user_data.get("role", "user")  # Default role is 'user'
                }
                return User(**filtered_data)  # Create User instance from filtered data

        except Exception as e:
            # Log an error for debugging purposes
            print(f"Error fetching user by email '{email}': {e}")

        # Return None if the user does not exist or an error occurred
        return None

    @staticmethod
    def get_by_id(user_id):
        response = supabase.table("users").select("*").eq("id", user_id).execute()
        if response.data:
            user_data = response.data[0]
            # Filter out the 'password' field (if it exists)
            filtered_data = {
                "id": user_data["id"],
                "name": user_data["name"],
                "email": user_data["email"],
                "password_hash": user_data["password_hash"],
                "role": user_data.get("role", "user")  # Default role is 'user'
            }
            return User(**filtered_data)
        return None

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Fetch user from Supabase
        user = User.get_by_email(email)

        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password!', 'danger')

    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        print(f"Received signup data: {name}, {email}")

        # Check if email exists
        response = supabase.table('users').select('email').eq('email', email).execute()
        if response.data:
            flash('Email already exists.', 'danger')
            return redirect(url_for('signup'))

        # Hash the password
        password_hash = generate_password_hash(password)

        # Insert user
        try:
            response = supabase.table('users').insert({
                'name': name,
                'email': email,
                'password_hash': password_hash,
                'role': 'user'
            }).execute()

            if response.data:
                flash('Account created successfully! Please log in.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Failed to create an account. Try again.', 'danger')

        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')

    return render_template('signup.html')

# Protected Route (Dashboard)
@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch user details from the database (if needed)
    if current_user.is_authenticated:
        user_id = current_user.id
        # Fetch additional user data from Supabase if necessary
        response = supabase.table('users').select('*').eq('id', user_id).execute()
        user_data = response.data[0] if response.data else None

        if user_data:
            # Pass user data to the template
            return render_template('dashboard.html', user=user_data)
        else:
            flash('User data not found.', 'danger')
            return redirect(url_for('login'))
    else:
        flash('You need to log in to access the dashboard.', 'danger')
        return redirect(url_for('login'))

def get_user_activities():
    response = supabase.table("activity_log").select("*").execute()
    activities = response.data  # returns dictionaries
    return activities

@app.route('/activitylog')
def show_activities():
    activities = get_user_activities()
    return render_template('activitylog.html', activities=activities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)