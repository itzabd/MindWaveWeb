import os
from supabase import create_client, Client
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from functools import wraps
from flask import flash, redirect, url_for
from flask_login import current_user
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set maximum content length (file size limit)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB


# Set secret key from environment variables
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://okhrguykcmeaakxclkbl.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9raHJndXlrY21lYWFreGNsa2JsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAwNTM0NzMsImV4cCI6MjA1NTYyOTQ3M30.XytMV4yJ5GMmhD_53E4rAByqBSY8GcqD1C0B3IWBjh8")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials! Check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
#====


app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # 30-minute session expiry

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, name, email, password_hash, email_verified=False, role_id=None):
        self.id = id
        self.username = username
        self.name = name
        self.email = email
        self.password_hash = password_hash
        self.email_verified = email_verified
        self.role_id = role_id  # Add role_id field

    def __repr__(self):
        return f"<User {self.name}>"

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def has_permission(self, permission_name):
        # Fetch permissions for the user's role
        response = supabase.table('permissions').select('permission_name').eq('role_id', self.role_id).execute()
        permissions = [perm['permission_name'] for perm in response.data]
        return permission_name in permissions

    def get_permissions(self):
        # Fetch all permissions for the user's role
        response = supabase.table('permissions').select('permission_name').eq('role_id', self.role_id).execute()
        return [perm['permission_name'] for perm in response.data] if response.data else []

    def get_role_name(self):
        # Fetch the role name for the user
        response = supabase.table('roles').select('name').eq('id', self.role_id).execute()
        if response.data:
            return response.data[0]['name']
        return None

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
                    "username": user_data["username"],
                    "name": user_data["name"],
                    "email": user_data["email"],
                    "password_hash": user_data["password_hash"],
                    "email_verified": user_data.get("email_verified", False),
                    "role_id": user_data.get("role_id")  # Fetch role_id
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
                "username": user_data["username"],
                "name": user_data["name"],
                "email": user_data["email"],
                "password_hash": user_data["password_hash"],
                "email_verified": user_data.get("email_verified", False),
                "role_id": user_data.get("role_id")  # Fetch role_id
            }
            return User(**filtered_data)
        return None

    @staticmethod
    def log_session(user_id):
        """Log user login session in the database."""
        try:
            response = supabase.table('user_logins').insert({
                'userid': user_id,
                'login_timestamp': 'now()',  # Automatically set the current timestamp
                'ip_address': request.remote_addr,  # Log the user's IP address
                'user_agent': request.headers.get('User-Agent')  # Log the user's browser/device info
            }).execute()
            return response.data
        except Exception as e:
            print(f"Error logging session: {e}")
            return None

    def set_remember_me_token(self, token):
        """Store the 'Remember Me' token in the database."""
        try:
            response = supabase.table('users').update({
                'remember_me_token': token
            }).eq('id', self.id).execute()
            return response.data
        except Exception as e:
            print(f"Error setting remember me token: {e}")
            return None

    @staticmethod
    def get_by_remember_me_token(token):
        """Fetch a user by their 'Remember Me' token."""
        try:
            response = supabase.table('users').select('*').eq('remember_me_token', token).execute()
            if response.data:
                user_data = response.data[0]
                return User(**user_data)
        except Exception as e:
            print(f"Error fetching user by remember me token: {e}")
        return None


# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

# Home route
@app.route('/')
def home():
    if current_user.is_authenticated:
        # Redirect based on user role
        role_name = current_user.get_role_name()
        if role_name == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif role_name == 'user':
            return redirect(url_for('user_dashboard'))
        elif role_name == 'guest':
            return redirect(url_for('guest_dashboard'))
        else:
            return redirect(url_for('dashboard'))

    return render_template('index.html')


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember_me = 'remember_me' in request.form  # Check if "Remember Me" is selected

        user = User.get_by_email(email)

        if user and user.check_password(password):
            login_user(user, remember=remember_me)  # Enable remember_me
            session.permanent = True  # Enable session expiry
            User.log_session(user.id)
            flash('Logged in successfully!', 'success')

            # Redirect based on user role
            role_name = user.get_role_name()
            if role_name == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif role_name == 'user':
                return redirect(url_for('user_dashboard'))
            elif role_name == 'guest':
                return redirect(url_for('guest_dashboard'))
            else:
                return redirect(url_for('dashboard'))

        else:
            flash('Invalid email or password!', 'danger')

    return render_template('login.html')



# Logout Route
@app.route('/logout')
@login_required
def logout():
    # Update the logout_timestamp for the current session
    try:
        response = supabase.table('user_logins').update({
            'logout_timestamp': 'now()'  # Use 'now()' to set the current timestamp
        }).eq('userid', current_user.id).is_('logout_timestamp', 'NULL').execute()

        if response.data:
            flash('Logged out successfully!', 'success')
        else:
            flash('No active session found.', 'info')
    except Exception as e:
        flash(f"Error logging out: {str(e)}", 'danger')

    logout_user()  # Clear the Flask-Login session
    return redirect(url_for('home'))

# Signup Route..................................

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'demomindwaveweb@gmail.com'
app.config['MAIL_PASSWORD'] = 'qvpo pdfo xwyy juue' #App Password
mail = Mail(app)
from flask_mail import Message

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if email or username already exists
        response = supabase.table('users').select('email, username').or_(f'email.eq.{email},username.eq.{username}').execute()
        if response.data:
            flash('Email or username already exists.', 'danger')
            return redirect(url_for('signup'))

        # Hash the password
        password_hash = generate_password_hash(password)

        # Fetch the default role_id for 'user'
        role_response = supabase.table('roles').select('id').eq('name', 'user').execute()
        if not role_response.data:
            flash('Default role not found.', 'danger')
            return redirect(url_for('signup'))

        role_id = role_response.data[0]['id']

        # Insert user into the database
        try:
            response = supabase.table('users').insert({
                'username': username,
                'name': name,
                'email': email,
                'password_hash': password_hash,
                'email_verified': False,
                'role_id': role_id  # Assign default role
            }).execute()

            if response.data:
                flash('Account created successfully! Please check your email to verify your account.', 'success')

                # Generate email verification token
                token = generate_verification_token(email)
                verification_url = url_for('verify_email', token=token, _external=True)

                # Send verification email
                try:
                    msg = Message("Verify Your Email", sender="demomindwaveweb@gmail.com", recipients=[email])
                    msg.body = f"Click the link to verify your email: {verification_url}"
                    mail.send(msg)
                    flash('Verification email sent!', 'success')  # Optional: Confirmation flash
                except Exception as e:
                    flash(f"Failed to send email: {str(e)}", 'danger')
                    print(f"Error sending email: {str(e)}")  # Debugging log

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
    # Refresh user data from the database
    user = User.get_by_id(current_user.id)  # Fetch the latest user data
    role_name = user.get_role_name()
    if role_name == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif role_name == 'user':
        return redirect(url_for('user_dashboard'))
    elif role_name == 'guest':
        return redirect(url_for('guest_dashboard'))
    else:
        return redirect(url_for('dashboard'))

    # user = User.get_by_id(current_user.id)  # Fetch the latest user data
    # if user:
    #     return render_template('dashboard.html', user=user)  # Pass updated user
    # else:
    #     flash('User data not found.', 'danger')
    #     return redirect(url_for('login'))



#==========Gen and verify token======================
def generate_verification_token(email):
    serializer = URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(email, salt='email-verification')

def verify_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(app.secret_key)
    try:
        email = serializer.loads(token, salt='email-verification', max_age=expiration)
    except:
        return None
    return email
#===================email verification========================
@app.route('/verify_email/<token>')
def verify_email(token):
    email = verify_token(token)  # Decode the token
    if not email:
        flash('Invalid or expired token.', 'danger')
        return redirect(url_for('login'))

    # Check if email is already verified
    response = supabase.table('users').select('email_verified').eq('email', email).execute()
    if response.data and response.data[0]['email_verified']:
        flash('Email is already verified. You can log in.', 'info')
        return redirect(url_for('login'))

    # ✅ Update email_verified in Supabase
    update_response = supabase.table('users').update({'email_verified': True}).eq('email', email).execute()

    print(f"Email verification update response: {update_response}")  # Debugging log

    # ✅ Fetch updated user data
    user = User.get_by_email(email)
    if user:
        login_user(user, remember=True)  # 🔹 Refresh session properly
        flash('Email verified successfully! You can now log in.', 'success')
        return redirect(url_for('dashboard'))

    flash('Verification successful, but there was an issue loading your profile.', 'warning')
    return redirect(url_for('login'))


#resend verification
@app.route('/resend_verification', methods=['POST'])
def resend_verification():
    email = request.form['email']

    # ✅ Fetch latest user data to ensure email verification status is correct
    user = User.get_by_email(email)
    if user and user.email_verified:
        flash('Your email is already verified. You can log in.', 'info')
        return redirect(url_for('login'))

    if user:
        # Generate a new token and send verification email
        token = generate_verification_token(email)
        verification_url = url_for('verify_email', token=token, _external=True)
        try:
            msg = Message("Verify Your Email", sender="demomindwaveweb@gmail.com", recipients=[email])
            msg.body = f"Click the link to verify your email: {verification_url}"
            mail.send(msg)
            flash('A new verification email has been sent!', 'success')
        except Exception as e:
            flash(f"Failed to send email: {str(e)}", 'danger')
    else:
        flash("No account found with this email.", "danger")

    return redirect(url_for('login'))
#Admin access
def role_required(role_name):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Fetch the role_id for the given role_name
            role_response = supabase.table('roles').select('id').eq('name', role_name).execute()
            if not role_response.data:
                flash('Role not found.', 'danger')
                return redirect(url_for('home'))

            role_id = role_response.data[0]['id']

            # Check if the user has the required role
            if not current_user.is_authenticated or current_user.role_id != role_id:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('home'))

            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/user')
@login_required
@role_required('user')
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route('/guest')
@role_required('guest')
def guest_dashboard():
    return render_template('guest_dashboard.html')
#========functionality to manage user groups==========

@app.route('/add_user_to_group', methods=['POST'])
@login_required
@role_required('admin')
def add_user_to_group():
    user_id = request.form['user_id']
    group_id = request.form['group_id']

    try:
        supabase.table('user_groups').insert({'user_id': user_id, 'group_id': group_id}).execute()
        flash('User added to group successfully!', 'success')
    except Exception as e:
        flash(f"Error: {str(e)}", 'danger')

    return redirect(url_for('admin_dashboard'))

@app.route('/remove_user_from_group', methods=['POST'])
@login_required
@role_required('admin')
def remove_user_from_group():
    user_id = request.form['user_id']
    group_id = request.form['group_id']

    try:
        supabase.table('user_groups').delete().eq('user_id', user_id).eq('group_id', group_id).execute()
        flash('User removed from group successfully!', 'success')
    except Exception as e:
        flash(f"Error: {str(e)}", 'danger')

    return redirect(url_for('admin_dashboard'))
@app.route('/user/edit_profile', methods=['GET', 'POST'])
@login_required
@role_required('user')
def edit_profile():
    if request.method == 'POST':
        # Handle form submission to update the user's profile
        name = request.form.get('name')
        email = request.form.get('email')
        # Update the user's profile in the database
        response = supabase.table('users').update({
            'name': name,
            'email': email
        }).eq('id', current_user.id).execute()
        if response.data:
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('user_dashboard'))
        else:
            flash('Failed to update profile. Please try again.', 'danger')
    # Render the edit profile form
    return render_template('edit_profile.html', user=current_user)
@app.route('/admin/add_user', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def add_user():
    if request.method == 'POST':
        # Handle form submission to add a new user
        username = request.form.get('username')
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        role_id = request.form.get('role_id')

        # Hash the password
        password_hash = generate_password_hash(password)

        # Insert the new user into the database
        response = supabase.table('users').insert({
            'username': username,
            'name': name,
            'email': email,
            'password_hash': password_hash,
            'email_verified': False,
            'role_id': role_id
        }).execute()

        if response.data:
            flash('User added successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Failed to add user. Please try again.', 'danger')

    # Fetch roles for the dropdown
    roles_response = supabase.table('roles').select('*').execute()
    roles = roles_response.data if roles_response.data else []

    # Render the add user form
    return render_template('add_user.html', roles=roles)
@app.route('/admin/users')
@login_required
@role_required('admin')
def list_users():
    # Fetch all users from the database
    response = supabase.table('users').select('*').execute()
    users = response.data if response.data else []
    return render_template('list_users.html', users=users)
@app.route('/admin/roles', methods=['GET'])
@login_required
@role_required('admin')
def list_roles():
    # Fetch all roles from the database
    response = supabase.table('roles').select('*').execute()
    roles = response.data if response.data else []
    return render_template('list_roles.html', roles=roles)
@app.route('/admin/groups', methods=['GET'])
@login_required
@role_required('admin')
def list_groups():
    response = supabase.table('groups').select('*').execute()
    groups = response.data if response.data else []
    return render_template('list_groups.html', groups=groups)
@app.route('/admin/add_group', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def add_group():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        response = supabase.table('groups').insert({
            'name': name,
            'description': description
        }).execute()
        if response.data:
            flash('Group added successfully!', 'success')
            return redirect(url_for('list_groups'))
        else:
            flash('Failed to add group. Please try again.', 'danger')
    return render_template('add_group.html')
@app.route('/admin/assign_user_to_group', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def assign_user_to_group():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        group_id = request.form.get('group_id')
        response = supabase.table('user_groups').insert({
            'user_id': user_id,
            'group_id': group_id
        }).execute()
        if response.data:
            flash('User assigned to group successfully!', 'success')
            return redirect(url_for('list_groups'))
        else:
            flash('Failed to assign user to group. Please try again.', 'danger')

    # Fetch users and groups for the dropdowns
    users_response = supabase.table('users').select('*').execute()
    groups_response = supabase.table('groups').select('*').execute()
    users = users_response.data if users_response.data else []
    groups = groups_response.data if groups_response.data else []

    return render_template('assign_user_to_group.html', users=users, groups=groups)


@app.route('/admin/add_role', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def add_role():
    if request.method == 'POST':
        # Handle form submission to add a new role
        name = request.form.get('name')
        description = request.form.get('description')

        # Insert the new role into the database
        response = supabase.table('roles').insert({
            'name': name,
            'description': description
        }).execute()

        if response.data:
            flash('Role added successfully!', 'success')
            return redirect(url_for('list_roles'))
        else:
            flash('Failed to add role. Please try again.', 'danger')

    # Render the add role form
    return render_template('add_role.html')
@app.route('/admin/permissions', methods=['GET'])
@login_required
@role_required('admin')
def list_permissions():
    # Fetch all permissions from the database
    response = supabase.table('permissions').select('*').execute()
    permissions = response.data if response.data else []
    return render_template('list_permissions.html', permissions=permissions)

@app.route('/admin/add_permission', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def add_permission():
    if request.method == 'POST':
        # Handle form submission to add a new permission
        permission_name = request.form.get('permission_name')
        role_id = request.form.get('role_id')

        # Insert the new permission into the database
        response = supabase.table('permissions').insert({
            'permission_name': permission_name,
            'role_id': role_id
        }).execute()

        if response.data:
            flash('Permission added successfully!', 'success')
            return redirect(url_for('list_permissions'))
        else:
            flash('Failed to add permission. Please try again.', 'danger')

    # Fetch roles for the dropdown
    roles_response = supabase.table('roles').select('*').execute()
    roles = roles_response.data if roles_response.data else []

    # Render the add permission form
    return render_template('add_permission.html', roles=roles)
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@role_required('admin')
def delete_user(user_id):
    if not current_user.has_permission('delete_user'):
        flash('You do not have permission to delete users.', 'danger')
        return redirect(url_for('admin_dashboard'))

    response = supabase.table('users').delete().eq('id', user_id).execute()
    if response.data:
        flash('User deleted successfully!', 'success')
    else:
        flash('Failed to delete user. Please try again.', 'danger')
    return redirect(url_for('list_users'))
@app.route('/user/change_password', methods=['GET', 'POST'])
@login_required
@role_required('user')
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Verify the current password
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('change_password'))

        # Check if the new password and confirmation match
        if new_password != confirm_password:
            flash('New password and confirmation do not match.', 'danger')
            return redirect(url_for('change_password'))

        # Hash the new password
        new_password_hash = generate_password_hash(new_password)

        # Update the user's password in the database
        response = supabase.table('users').update({
            'password_hash': new_password_hash
        }).eq('id', current_user.id).execute()

        if response.data:
            flash('Password changed successfully!', 'success')
            return redirect(url_for('user_dashboard'))
        else:
            flash('Failed to change password. Please try again.', 'danger')

    return render_template('change_password.html')

#======Login Session ==========
@app.route('/sessions')
@login_required
def view_sessions():
    # Fetch active sessions for the current user
    response = supabase.table('user_logins').select('*').eq('userid', current_user.id).is_('logout_timestamp', 'NULL').execute()
    sessions = response.data if response.data else []
    return render_template('sessions.html', sessions=sessions)
@app.route('/logout_all_sessions', methods=['POST'])
@login_required
def logout_all_sessions():
    # Delete all sessions for the current user
    try:
        response = supabase.table('user_logins').delete().eq('userid', current_user.id).execute()
        if response.data:
            flash('Logged out of all sessions successfully!', 'success')
        else:
            flash('No active sessions found.', 'info')
    except Exception as e:
        flash(f"Error logging out of all sessions: {str(e)}", 'danger')

    return redirect(url_for('dashboard'))
@app.route('/logout_session/<int:session_id>', methods=['POST'])
@login_required
def logout_session(session_id):
    try:
        response = supabase.table('user_logins').delete().eq('id', session_id).eq('userid', current_user.id).execute()
        if response.data:
            flash('Session logged out successfully!', 'success')
        else:
            flash('Session not found.', 'info')
    except Exception as e:
        flash(f"Error logging out session: {str(e)}", 'danger')
    return redirect(url_for('view_sessions'))

#login session End=============
#Password reset feature
def generate_reset_token(email):
    serializer = URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(email, salt='password-reset')

def verify_reset_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(app.secret_key)
    try:
        email = serializer.loads(token, salt='password-reset', max_age=expiration)
    except:
        return None
    return email
@app.route('/request_password_reset', methods=['GET', 'POST'])
def request_password_reset():
    if request.method == 'POST':
        email = request.form['email']
        user = User.get_by_email(email)

        if user:
            # Generate a reset token
            token = generate_reset_token(email)
            reset_url = url_for('reset_password', token=token, _external=True)

            # Send reset email
            try:
                msg = Message("Password Reset Request", sender="demomindwaveweb@gmail.com", recipients=[email])
                msg.body = f"Click the link to reset your password: {reset_url}"
                mail.send(msg)
                flash('A password reset link has been sent to your email.', 'success')
            except Exception as e:
                flash(f"Failed to send email: {str(e)}", 'danger')
        else:
            flash("No account found with this email.", "danger")

    return render_template('request_password_reset.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token)
    if not email:
        flash('Invalid or expired token.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('reset_password', token=token))

        # Hash the new password
        password_hash = generate_password_hash(new_password)

        # Update the user's password in the database
        response = supabase.table('users').update({
            'password_hash': password_hash
        }).eq('email', email).execute()

        if response.data:
            flash('Your password has been reset successfully!', 'success')
            return redirect(url_for('login'))
        else:
            flash('Failed to reset password. Please try again.', 'danger')

    return render_template('reset_password.html', token=token)
#---------------------------
def plot():
    # Load and process dataset
    file_path = "dataforvisual/GGS_new.csv"
    df = pd.read_csv(file_path, delimiter=";")

    high_edu_df = df[df["edu_level"] == 1]
    gender_counts = high_edu_df["sex"].value_counts().sort_index()
    labels = {1: "Men", 2: "Women"}
    plt.switch_backend('Agg')
    plt.figure(figsize=(6, 4))
    plt.bar(gender_counts.index.map(labels), gender_counts.values, color=["blue", "pink"])
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Men & Women with High Education Level")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("static/gender_count.png")

@app.route("/graph")
def graph():
    plot()
    fig = "static/gender_count.png"
    return render_template('graph.html', fig=fig)

app.route('static/<path:filename>')
def statfile(filename):
    return send_file(f'static/{filename}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)