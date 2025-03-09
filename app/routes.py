# c:\proj1\app\routes.py
from flask import render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
import random
import pandas as pd
import math
import os
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText

from datetime import datetime
import random

# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gmail configuration (replace with real credentials)
GMAIL_EMAIL = "yourrealemail@gmail.com"
GMAIL_PASSWORD = "yourapppassword"

# MongoDB collections
# from database import get_users_collection, get_admins_collection, get_canal_data_collection
# users_collection = get_users_collection()
# admins_collection = get_admins_collection()
# canal_data_collection = get_canal_data_collection()

# Load CNN model
cnn_model = tf.keras.models.load_model('c:/proj1/waste_cnn_model.h5')
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
waste_types = ["plastic", "metal", "biomedical", "shoes"]

# Load CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # C:\proj1\app
CSV_PATH = os.path.join(BASE_DIR, 'static', 'water_quality.csv')  # C:\proj1\app\static\water_quality.csv
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please ensure it exists.")
df = pd.read_csv(CSV_PATH)

# Utility functions
def send_price_email(to_email, product_name, price):
    subject = f"{product_name} Price Quote"
    body = f"Thank you for your interest in {product_name}! The price is {price} per unit. Contact us for more details!"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = GMAIL_EMAIL
    msg['To'] = to_email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        return False

def estimate_d5(d0, temperature):
    decay_rate = 0.1 + (temperature / 100)  # Simplified without chlorophyll/ph
    return d0 * math.exp(-decay_rate * 5)

def calculate_bod(d0, d5):
    return d0 - d5 if d0 > d5 else 0

def predict_diseases(bod):
    if bod <= 3:
        category = "Low BOD (Clean Water)"
        health_risk = "Minimal"
        diseases = ["None (if free from other contaminants)"]
        solutions = ["Water is generally safe for drinking and recreation. Regular monitoring recommended."]
    elif 3 < bod <= 5:
        category = "Moderate BOD (Slightly Polluted Water)"
        health_risk = "Moderate"
        diseases = ["Gastroenteritis (Stomach Flu)", "Skin Infections & Rashes"]
        solutions = ["Boil water before drinking or use a water purifier. Treat with chlorine if needed.",
                     "Avoid prolonged skin contact; wash with clean water after exposure."]
    elif 5 < bod <= 10:
        category = "High BOD (Heavily Polluted Water)"
        health_risk = "High"
        diseases = ["Cholera", "Typhoid Fever", "Hepatitis A & E"]
        solutions = ["Seek medical help if symptoms like severe diarrhea appear. Use ORS and antibiotics (e.g., doxycycline) under doctor supervision.",
                     "Do not drink untreated water; boil or filter thoroughly.",
                     "Get vaccinated for Hepatitis A if possible; consult a doctor for symptoms like jaundice."]
    else:
        category = "Very High BOD (Severely Polluted Water)"
        health_risk = "Extremely High (Life-threatening)"
        diseases = ["Dysentery", "Polio", "Leptospirosis", "Arsenic Poisoning", "Blue Baby Syndrome"]
        solutions = ["Immediate medical attention for bloody diarrhea or fever. Antibiotics (e.g., metronidazole) may be needed.",
                     "Vaccinate against polio; avoid all contact with water.",
                     "Seek treatment for fever or muscle aches; avoid floodwater exposure.",
                     "Test for heavy metals; consult a specialist for long-term exposure symptoms.",
                     "Use nitrate-free water for infants; seek pediatric care if blue skin appears."]
    return category, health_risk, diseases, solutions

# Authentication (assuming auth.py exists; otherwise, define here)
from auth import authenticate, is_authenticated

# Decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
def init_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if authenticate(username, password, users_collection):
                session['user'] = {'username': username}
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid user credentials. Please sign up or try again.')
                return redirect(url_for('signup'))
        signup_url = url_for('signup')
        admin_login_url = url_for('admin_login')
        return render_template('login.html', signup_url=signup_url, admin_login_url=admin_login_url)

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if users_collection.find_one({'username': username}):
                flash('Username already exists. Please log in.')
                return redirect(url_for('login'))
            users_collection.insert_one({'username': username, 'password': password})
            flash('Signup successful! Please log in.')
            return redirect(url_for('login'))
        return render_template('signup.html')

    @app.route('/admin_login', methods=['GET', 'POST'])
    def admin_login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if authenticate(username, password, admins_collection):
                session['admin'] = {'username': username}  # Use 'admin' key for admin session
                return redirect(url_for('admin_home'))
            else:
                flash('Invalid admin credentials. Please sign up or try again.')
                return redirect(url_for('admin_signup'))
        admin_signup_url = url_for('admin_signup')
        return render_template('admin_login.html', admin_signup_url=admin_signup_url)

    @app.route('/admin_signup', methods=['GET', 'POST'])
    def admin_signup():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if admins_collection.find_one({'username': username}):
                flash('Admin username already exists. Please log in.')
                return redirect(url_for('admin_login'))
            admins_collection.insert_one({'username': username, 'password': password})
            flash('Admin signup successful! Please log in.')
            return redirect(url_for('admin_login'))
        return render_template('admin_signup.html')

    @app.route('/logout')
    def logout():
        session.pop('user', None)
        session.pop('admin', None)
        return redirect(url_for('login'))

    @app.route('/buy_sell')
    @login_required
    def buy_sell():
        return render_template('buy_sell.html')

    @app.route('/sell', methods=['GET', 'POST'])
    @login_required
    def sell():
        if request.method == 'POST':
            return render_template('sell.html', show_message=True)
        return render_template('sell.html', show_message=False)

    @app.route('/buy', methods=['GET', 'POST'])
    @login_required
    def buy():
        if request.method == 'POST':
            waste_kg = request.form.get('waste_kg', type=int)
            if waste_kg is None or waste_kg < 0:
                message = "Please enter a valid amount of waste!"
            else:
                message = f"Simulated purchase of {waste_kg} kg of waste!"
            return render_template('buy.html', message=message)
        return render_template('buy.html')

    @app.route('/dashboard')
    @login_required
    def dashboard():
        data = list(users_collection.find({}, {"_id": 0}))
        today = datetime.now().strftime("%Y-%m-%d")
        return render_template('dashboard.html', today=today)

    @app.route('/upload_waste_images', methods=['POST'])
    @login_required
    def upload_waste_images():
        if 'images' not in request.files or not request.form.get('date') or not request.form.get('canal'):
            flash("Please provide date, canal, and at least one image!")
            return redirect(url_for('dashboard'))
        
        date_str = request.form['date']
        canal_name = request.form['canal']
        uploaded_files = request.files.getlist("images")
        
        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            flash("No images uploaded!")
            return redirect(url_for('dashboard'))

        waste_counts = {"plastic": 0, "metal": 0, "biomedical": 0, "shoes": 0}
        for file in uploaded_files:
            if file.filename != '':
                image = Image.open(file.stream).resize((224, 224))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                prediction = cnn_model.predict(image_array)[0]
                for i, prob in enumerate(prediction):
                    if prob > 0.5:
                        waste_counts[waste_types[i]] += 1

        canal_data_collection.update_one(
            {"date": date_str, "canal": canal_name},
            {"$set": {"waste_counts": waste_counts}},
            upsert=True
        )
        flash(f"Waste images for {canal_name} on {date_str} uploaded and classified successfully!")
        return redirect(url_for('dashboard'))

    @app.route('/product1', methods=['GET', 'POST'])
    def product1():
        if request.method == 'POST':
            email = request.form.get('email')
            if email and '@' in email:
                if send_price_email(email, "AquaClense 1.0", "$100,000"):
                    flash(f"Price details for AquaClense 1.0 sent to {email}!")
                else:
                    flash("Failed to send price details. Please try again.")
                return redirect(url_for('product1'))
            else:
                flash("Please enter a valid email address!")
        return render_template('product1.html')

    @app.route('/product2', methods=['GET', 'POST'])
    def product2():
        if request.method == 'POST':
            email = request.form.get('email')
            if email and '@' in email:
                if send_price_email(email, "AquaClense 2.0", "$130,000"):
                    flash(f"Price details for AquaClense 2.0 sent to {email}!")
                else:
                    flash("Failed to send price details. Please try again.")
                return redirect(url_for('product2'))
            else:
                flash("Please enter a valid email address!")
        return render_template('product2.html')

    @app.route('/product3', methods=['GET', 'POST'])
    def product3():
        if request.method == 'POST':
            email = request.form.get('email')
            if email and '@' in email:
                if send_price_email(email, "AquaClense 3.0", "90,000 INR (starting price, depends on canal size)"):
                    flash(f"Price details for AquaClense 3.0 sent to {email}!")
                else:
                    flash("Failed to send price details. Please try again.")
                return redirect(url_for('product3'))
            else:
                flash("Please enter a valid email address!")
        return render_template('product3.html')

    @app.route('/product4', methods=['GET', 'POST'])
    def product4():
        if request.method == 'POST':
            email = request.form.get('email')
            if email and '@' in email:
                if send_price_email(email, "AquaClense 4.0", "1.7 lakh INR (depending on canal length)"):
                    flash(f"Price details for AquaClense 4.0 sent to {email}!")
                else:
                    flash("Failed to send price details. Please try again.")
                return redirect(url_for('product4'))
            else:
                flash("Please enter a valid email address!")
        return render_template('product4.html')

    @app.route('/reports', methods=['GET', 'POST'])
    @login_required
    def reports():
        if request.method == 'POST':
            selected_date = request.form['date']
            canals = ["Canal A", "Canal B", "Canal C"]
            canal_data = {}
            for canal in canals:
                data = canal_data_collection.find_one({"date": selected_date, "canal": canal})
                canal_data[canal] = data["waste_counts"] if data and "waste_counts" in data else {}
            return render_template('reports.html', date=selected_date, canal_data=canal_data)
        return render_template('select_date_canal.html')

    @app.route('/recycling_measures')
    @login_required
    def recycling_measures():
        return render_template('recycling_measures.html')

    @app.route('/get_waste_data/<date>/<canal>', methods=['GET'])
    @login_required
    def get_waste_data(date, canal):
        report = canal_data_collection.find_one({"date": date, "canal": canal})
        if report and "waste_counts" in report:
            return jsonify(report["waste_counts"])
        return jsonify({}), 404

    @app.route('/admin_home')
    @admin_required
    def admin_home():
        data = list(admins_collection.find({}, {"_id": 0}))
        return render_template('admin_home.html')



    @app.route('/wdp', methods=['GET', 'POST'])
    @login_required
    def wdp():
        logger.debug("Entering /wdp route")
        today_date = datetime.now().strftime("%Y-%m-%d")  # Default to today

        if not os.path.exists(CSV_PATH):
            logger.error(f"CSV file not found at {CSV_PATH}")
            flash(f"Error: CSV file not found at {CSV_PATH}")
            return render_template('wdp.html', today_date=today_date)

        try:
            logger.debug(f"Loading CSV from {CSV_PATH}")
            if df.empty:
                logger.error("CSV file is empty")
                flash("Error: CSV file is empty")
                return render_template('wdp.html', today_date=today_date)

            selected_date = request.form.get('selected_date', today_date) if request.method == 'POST' else today_date
            logger.debug(f"Selected date: {selected_date}")

            # Seed random with the selected date for consistent randomness per date
            date_seed = int(''.join(filter(str.isdigit, selected_date)))  # e.g., "2025-03-08" -> 20250308
            random.seed(date_seed)
            
            logger.debug(f"CSV columns: {df.columns.tolist()}")
            # Randomly sample one row with date-specific seed
            random_data = df.sample(n=1, random_state=date_seed).iloc[0]
            logger.debug(f"Randomly selected row for {selected_date}: {random_data.to_dict()}")

            # Extract data using your CSV's column names
            sample_id = random_data.get("Sample ID", "N/A")
            ph = float(random_data.get("pH", 7.0)) if not pd.isna(random_data.get("pH")) else 7.0
            temperature = float(random_data.get("Temperature (°C)", 20.0)) if not pd.isna(random_data.get("Temperature (°C)")) else 20.0
            turbidity = float(random_data.get("Turbidity (NTU)", 0.0)) if not pd.isna(random_data.get("Turbidity (NTU)")) else 0.0
            d0 = float(random_data.get("Dissolved Oxygen (mg/L)", 0.0)) if not pd.isna(random_data.get("Dissolved Oxygen (mg/L)")) else 0.0
            conductivity = float(random_data.get("Conductivity (µS/cm)", 0.0)) if not pd.isna(random_data.get("Conductivity (µS/cm)")) else 0.0

            logger.debug(f"Extracted: sample_id={sample_id}, ph={ph}, temp={temperature}, turbidity={turbidity}, d0={d0}, conductivity={conductivity}")
            
            if d0 <= 0:
                logger.error("Invalid dissolved oxygen value (D0 <= 0)")
                flash("Error: Dissolved oxygen value is invalid (must be > 0)")
                return render_template('wdp.html',
                                    selected_date=selected_date,
                                    today_date=today_date,
                                    sample_id=sample_id,
                                    d0=d0,
                                    d5=0.0,
                                    temperature=temperature,
                                    turbidity=turbidity,
                                    ph=ph,
                                    conductivity=conductivity,
                                    bod=0.0,
                                    category="Invalid Data",
                                    health_risk="N/A",
                                    diseases=["N/A"],
                                    solutions=["Check data source for valid DO values."])

            d5 = estimate_d5(d0, temperature)
            bod = calculate_bod(d0, d5)
            category, health_risk, diseases, solutions = predict_diseases(bod)

            return render_template('wdp.html', 
                                selected_date=selected_date,
                                today_date=today_date,
                                sample_id=sample_id,
                                d0=d0, 
                                d5=d5, 
                                temperature=temperature,
                                turbidity=turbidity,
                                ph=ph,
                                conductivity=conductivity,
                                bod=bod, 
                                category=category, 
                                health_risk=health_risk, 
                                diseases=diseases, 
                                solutions=solutions)
        except Exception as e:
            logger.error(f"Error in /wdp: {str(e)}")
            flash(f"Error processing water quality data: {str(e)}")
            return render_template('wdp.html', today_date=today_date)