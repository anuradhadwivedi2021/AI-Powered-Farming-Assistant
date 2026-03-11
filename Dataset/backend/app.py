# ================================
# AI Farming Assistant - Flask API
# ================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import requests
import os
import sys
from dotenv import load_dotenv

# .env file load karo
load_dotenv()

app = Flask(__name__)
CORS(app)

# ---- Model Load Karo ----
print("🌾 Loading AI Model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'crop_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        crop_model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load error: {e}")
    crop_model = None

# ================================
# HOME ROUTE
# ================================
@app.route('/')
def home():
    return jsonify({
        'message': '🌾 AI Farming Assistant API is Running!',
        'version': '1.0',
        'endpoints': {
            'crop_recommendation': '/api/recommend-crop',
            'weather': '/api/weather?city=Pune',
            'history': '/api/history'
        }
    })


# ================================
# CROP RECOMMENDATION API
# ================================
@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    try:
        data = request.json

        required = ['nitrogen', 'phosphorus', 'potassium',
                   'temperature', 'humidity', 'ph', 'rainfall']

        for field in required:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} field missing!'
                }), 400

        features = pd.DataFrame([[
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        predicted_crop = crop_model.predict(features)[0]

        fertilizer = get_fertilizer_suggestion(
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium'])
        )

        tips = get_farming_tips(predicted_crop)

        return jsonify({
            'success': True,
            'recommended_crop': predicted_crop.upper(),
            'fertilizer': fertilizer,
            'farming_tips': tips,
            'message': f'Best crop for your soil is {predicted_crop.upper()}!'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ================================
# WEATHER API
# ================================
@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city', 'Pune')
    api_key = os.getenv('WEATHER_API_KEY', 'd42752eb66da15596fb238312ad4c712')

    if not api_key:
        return jsonify({
            'success': True,
            'city': city,
            'temperature': 28.5,
            'humidity': 75,
            'description': 'Partly cloudy',
            'rainfall': 0,
            'note': 'Demo data - Add API key in .env file'
        })

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params)
        weather = response.json()

        if response.status_code == 200:
            return jsonify({
                'success': True,
                'city': city,
                'temperature': weather['main']['temp'],
                'humidity': weather['main']['humidity'],
                'description': weather['weather'][0]['description'],
                'rainfall': weather.get('rain', {}).get('1h', 0),
                'wind_speed': weather['wind']['speed']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'City not found'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ================================
# HISTORY API
# ================================
@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        import sqlite3
        conn = sqlite3.connect('farming.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_name TEXT,
                location TEXT,
                recommended_crop TEXT,
                fertilizer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            SELECT * FROM recommendations
            ORDER BY created_at DESC LIMIT 10
        ''')
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'id': row['id'],
                'farmer_name': row['farmer_name'],
                'location': row['location'],
                'recommended_crop': row['recommended_crop'],
                'fertilizer': row['fertilizer'],
                'date': row['created_at']
            })

        return jsonify({
            'success': True,
            'history': history
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ================================
# HELPER FUNCTIONS
# ================================
def get_fertilizer_suggestion(N, P, K):
    if N < 20:
        return "Urea fertilizer use karo - Nitrogen bahut kam hai"
    elif N < 40:
        return "Thoda Urea add karo - Nitrogen thoda kam hai"
    elif P < 20:
        return "DAP fertilizer use karo - Phosphorus kam hai"
    elif K < 20:
        return "MOP (Potash) fertilizer use karo - Potassium kam hai"
    elif N > 80 and P > 60 and K > 60:
        return "Soil nutrients balanced hain - Organic compost use karo"
    else:
        return "NPK 10:26:26 balanced fertilizer use karo"


def get_farming_tips(crop):
    tips = {
        'rice': [
            'Rice ke liye 20-27C temperature best hai',
            'Zyada paani chahiye - proper irrigation rakho',
            'Transplanting ke 20 din baad fertilizer do'
        ],
        'maize': [
            'Maize ke liye achhi drainage wali soil chahiye',
            'Har 7-10 din mein paani do',
            'Kharif season mein best grow hota hai'
        ],
        'wheat': [
            'Wheat ke liye 15-20C temperature best hai',
            'Rabi season crop hai - October-November mein bao',
            'Sandy loam soil best hai'
        ],
        'cotton': [
            'Cotton ko zyada sunlight chahiye',
            'Well-drained black soil best hai',
            'Bollworm se bachao - regular check karo'
        ]
    }
    return tips.get(crop.lower(), [
        f'{crop.upper()} ke liye proper irrigation rakho',
        'Regular soil testing karwao',
        'Pesticides sahi matra mein use karo'
    ])


# ================================
# RUN SERVER
# ================================
if __name__ == '__main__':
    print("🚀 Starting AI Farming Assistant Server...")
    print("🌐 Open browser: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
