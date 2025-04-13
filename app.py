from flask import Flask, render_template, request
from keras.models import load_model
from paddleocr import PaddleOCR
import numpy as np
import re
import requests

from PIL import Image

app = Flask(__name__)


ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Helper function
def extract_single_product_details(image_np):
    image = Image.fromarray(image_np)
    width, height = image.size
    brand_box_top, brand_box_bottom = int(0.05 * height), int(0.35 * height)
    weight_box_top, weight_box_bottom = int(0.55 * height), int(0.95 * height)

    cropped_brand = image_np[brand_box_top:brand_box_bottom, :]
    cropped_weight = image_np[weight_box_top:weight_box_bottom, :]

    brand_res = ocr.ocr(cropped_brand, cls=True)
    weight_res = ocr.ocr(cropped_weight, cls=True)

    brand_text = " ".join([line[1][0] for line in brand_res[0]]) if brand_res else ""
    weight_text = " ".join([line[1][0] for line in weight_res[0]]) if weight_res else ""

    brands = ["Bikaji", "VADAI","Good Day", "Haldiram", "Dettol", "Fortune", "BRU", "BEARDO", "PERK Mini Treats", "MAGGI", "KitKat", "Kissan", "Del Monte", "Bourn"]
    brand_pattern = re.compile(r'|'.join(brands), re.IGNORECASE)
    brand_match = brand_pattern.search(brand_text)
    brand_name = brand_match.group(0) if brand_match else 'Brand not found'

    combined_res = brand_text + " " + weight_text
    weight_match = re.search(r'NET (?:WEIGHT|WT|QUANTITY|Wt):?\s*([\d.]+)\s*(kg|g)', combined_res, re.IGNORECASE)
    net_weight = f"{weight_match.group(1)} {weight_match.group(2)}" if weight_match else 'Net Weight not found'

    return brand_name, net_weight

@app.route('/smartscan', methods=['GET', 'POST'])
def smartscan():
    brand_name = None
    net_weight = None

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file:
                image = Image.open(file).convert('RGB')
                image_np = np.array(image)
                brand_name, net_weight = extract_single_product_details(image_np)

    return render_template('smartscan.html', brand=brand_name, weight=net_weight)



# Load the fruit classification model
fruit_model = load_model('fruit_classifier_model.h5')

# Class labels mapping
fruit_class_mapping = {
    0: "Fresh Apples",
    1: "Fresh Bananas",
    2: "Fresh Oranges",
    3: "Rotten Apples",
    4: "Rotten Bananas",
    5: "Rotten Oranges"
}

# Helper functions
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_fruit(image):
    processed_image = preprocess_image(image)
    prediction = fruit_model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    return fruit_class_mapping[predicted_class_index]


LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

emoji_map = {
    "Plant": "🌱", "Blight": "🦠", "Fungicide": "🧪", "Soil": "🌿", "Temperature": "🌡",
    "Pest": "🐛", "Disease": "⚠️", "Leaves": "🍃", "Sunlight": "☀️", "Harvest": "🌾",
    "Prevention": "🛑", "Yield": "📈", "Spray": "🚿", "Weather": "⛅", "Resistant": "🛡",
    "Organic": "🌍", "Fertilizer": "🪴", "Air circulation": "🌬", "Chemical": "🧬",
    "Treatment": "💊", "Solution": "🔬", "Monitor": "📊", "Farm": "🚜", "Crops": "🌾",
    "Grow": "🌻"
}

def format_response(response_text):
    for word, emoji in emoji_map.items():
        response_text = re.sub(rf'\b{word}\b', f"{word} {emoji}", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r'(\d+\.)', r'🔹 \1', response_text)
    response_text = response_text.replace("•", "✅")
    return response_text

def get_ai_response(user_query):
    payload = {
        "model": "llama-3.1-natural-farmer",
        "messages": [
            {"role": "system", "content": "You are an AI assistant that provides clear, engaging answers with emojis and give answers professionally in English."},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.7,
        "top_p": 1
    }
    
    response = requests.post(LM_STUDIO_URL, json=payload)
    if response.status_code == 200:
        assistant_reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
        return format_response(assistant_reply)
    return f"❌ Error fetching AI response: {response.status_code} - {response.text}"


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fruitvision', methods=['GET', 'POST'])
def fruitvision():
    prediction_result = None
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file:
                image = Image.open(image_file)
                prediction_result = predict_fruit(image)
    return render_template('fruitvision.html', prediction=prediction_result)

@app.route('/krishiai', methods=['GET', 'POST'])
def krishiai():
    ai_response = ""
    if request.method == 'POST':
        city = request.form['city']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        crop = request.form['crop']
        user_query = f"I live have a farm in {city} and the temperature is {temperature}°C and the humidity is {humidity}%. I have {crop} crop. how to make higher yield in this and which diseases the crop can have how to cure them and which fertilizer to be used?"
        ai_response = get_ai_response(user_query)
    return render_template('krishiai.html', response=ai_response)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)

