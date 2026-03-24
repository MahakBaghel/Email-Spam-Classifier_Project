# =============================
# IMPORT LIBRARIES
# =============================
from flask import Flask, render_template, request, redirect, session, jsonify, flash,send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from datetime import date
import imaplib
import email 
from email.header import decode_header

nltk.download('stopwords')

# =============================
# APP CONFIG
# =============================
app = Flask(__name__)
app.secret_key = "secretkey"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# =============================
# DATABASE MODELS
# =============================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    mobile = db.Column(db.String(15))
    address = db.Column(db.Text)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    message = db.Column(db.Text)
    result = db.Column(db.String(20))
date = db.Column(db.Date, default=date.today)
# =============================
# LOAD MODEL
# =============================
model = pickle.load(open(r"D:\Email_Spam_Classifier_Project\model\model.pkl", "rb"))
vectorizer = pickle.load(open(r"D:\Email_Spam_Classifier_Project\model\vectorizer.pkl", "rb"))

# =============================
# TEXT CLEANING
# =============================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# =============================
# HOME (PROTECTED)
# =============================
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template("index.html")

# =============================
# REGISTER
# =============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        mobile = request.form['mobile']
        address = request.form['address']
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        # Check duplicate username or email
        user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if user:
            flash("User already exists with this username or email")
            return redirect('/login')

        new_user = User(
            fullname=fullname,
            email=email,
            mobile=mobile,
            address=address,
            username=username,
            password=password
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful")
        return redirect('/login')

    return render_template("register.html")

# =============================
# LOGIN
# =============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        # User not found
        if not user:
            flash("Candidate not found. Please register.")
            return redirect('/register')

        # Wrong password
        if not check_password_hash(user.password, password):
            flash("Incorrect password")
            return redirect('/login')

        session['user_id'] = user.id
        return redirect('/')

    return render_template("login.html")

# =============================
# LOGOUT
# =============================
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/login')

# =============================
# RESET PASSWORD
# =============================
@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        username = request.form['username']
        new_password = generate_password_hash(request.form['password'])

        user = User.query.filter_by(username=username).first()

        if not user:
            flash("User not found")
            return redirect('/register')

        user.password = new_password
        db.session.commit()

        flash("Password updated")
        return redirect('/login')

    return render_template("reset.html")

# =============================
# PREDICT (HTML FORM)
# =============================
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect('/login')

    message = request.form['message']

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vector)[0]

    prediction = "Spam" if result == 1 else "Not Spam"

    # Save history
    entry = Prediction(
        user_id=session['user_id'],
        message=message,
        result=prediction,
        date=date.utcnow()
    )
    db.session.add(entry)
    db.session.commit()

    return render_template("index.html", prediction=prediction)

# =============================
# API PREDICT
# =============================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    message = data.get("message", "")

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vector)[0]

    return jsonify({
        "prediction": "Spam" if result == 1 else "Not Spam"
    })

# =============================
# HISTORY
# =============================
@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect('/login')

    data = Prediction.query.filter_by(user_id=session['user_id']).all()
    return render_template("history.html", data=data)

# =============================
# ANALYTICS
# =============================
@app.route('/analytics')
def analytics():
    spam = Prediction.query.filter_by(result="Spam").count()
    ham = Prediction.query.filter_by(result="Not Spam").count()

    return render_template("analytics.html", spam=spam, ham=ham)

# =============================
# ADMIN PANEL
# =============================
@app.route('/admin')
def admin():
    # Total users
    total_users = User.query.count()

    # Total predictions
    total_predictions = Prediction.query.count()

    # Spam & Not Spam count
    spam_count = Prediction.query.filter_by(result="Spam").count()
    ham_count = Prediction.query.filter_by(result="Not Spam").count()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        spam=spam_count,
        ham=ham_count
    )
# UPLOAD PANEL
@app.route('/upload')
def upload_page():
    return render_template("upload.html")


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files['file']
    df = pd.read_csv(file)

    results = []

    for msg in df['message']:
        cleaned = clean_text(str(msg))
        vector = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vector)[0]

        results.append("Spam" if pred == 1 else "Not Spam")

    df['Result'] = results

    df.to_csv("output.csv", index=False)

    return "CSV Processed Successfully!"

#GMAIL PENAL

def fetch_gmail():

    username = "your_email@gmail.com"
    password = "your_app_password"

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)

    mail.select("inbox")

    status, messages = mail.search(None, "ALL")

    email_list = []

    for num in messages[0].split()[-10:]:

        status, data = mail.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(data[0][1])

        subject = msg["subject"]

        if subject:
            email_list.append(subject)

    mail.logout()

    return email_list


# -------------------------------
# GMAIL ROUTE
# -------------------------------
@app.route('/gmail')
def gmail():

    try:
        emails = fetch_gmail()

        results = []

        for e in emails:

            cleaned = clean_text(str(e))
            vector = vectorizer.transform([cleaned]).toarray()
            pred = model.predict(vector)[0]

            result = "Spam" if pred == 1 else "Not Spam"

            results.append({
                "email": e,
                "result": result
            })

        return render_template("gmail.html", data=results)

    except Exception as e:
        return f"Gmail Error: {str(e)}"

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)