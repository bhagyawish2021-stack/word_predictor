"""
Next Word Predictor - Flask Backend
A modern AI-powered typing assistant for real-time word prediction
"""

from flask import Flask, render_template, request, jsonify
import predictor
import logging
import os


NextWordPredictor = predictor.NextWordPredictor
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = None


def init_predictor():
    """Initialize the predictor model"""
    global predictor
    model_path = os.path.join(os.path.dirname(__file__), 'predictor', 'model.pkl')
    
    if os.path.exists(model_path):
        logger.info("Loading existing model...")
        predictor = NextWordPredictor.load(model_path)
    else:
        logger.info("Training new model...")
        predictor = NextWordPredictor()
        corpus_path = os.path.join(os.path.dirname(__file__), 'dataset', 'corpus.txt')
        predictor.train(corpus_path)
        predictor.save(model_path)
    
    logger.info("Predictor initialized successfully")


@app.route('/')
def index():
    """Render the main application"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict next words based on input text
    
    Request JSON:
    {
        "text": "I am going to"
    }
    
    Response JSON:
    {
        "suggestions": [
            {"word": "school", "confidence": 0.85},
            {"word": "sleep", "confidence": 0.78},
            ...
        ],
        "input_text": "I am going to"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'suggestions': [],
                'input_text': text,
                'error': None
            })
        
        suggestions = predictor.predict(text, num_predictions=5)
        
        return jsonify({
            'suggestions': suggestions,
            'input_text': text,
            'error': None
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'suggestions': [],
            'input_text': '',
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get predictor statistics"""
    try:
        stats_data = {
            'model_type': 'N-gram Language Model',
            'vocabulary_size': len(predictor.vocabulary),
            'total_tokens': predictor.total_tokens,
            'training_status': 'ready'
        }
        return jsonify(stats_data)
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None
    })


@app.before_request
def before_request():
    """Initialize predictor before first request"""
    global predictor
    if predictor is None:
        init_predictor()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
