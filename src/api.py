from flask import Flask, request, jsonify
from src.models import load_model

app = Flask(__name__)
model = load_model('models/lr_tfidf.pkl')
cache = {}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text','')
    if text in cache:
        return jsonify({'label': cache[text], 'cached': True})
    pred = model.predict([text])[0]
    cache[text] = pred
    return jsonify({'label': str(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
