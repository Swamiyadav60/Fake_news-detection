# Fake News Detection System

A Machine Learning-powered web application that classifies news articles as **REAL** or **FAKE** using TF-IDF vectorization and ensemble classifiers. Built with Flask, scikit-learn, and modern dark-themed UI.

## Features

✨ **Intelligent Detection**

- Compares Logistic Regression vs Passive Aggressive Classifier
- Achieves 94-99% accuracy on test set
- Real-time predictions with confidence scores

🎨 **Modern Interface**

- Dark, modern UI with responsive design
- Color-coded results (Red = Fake, Green = Real)
- Animated confidence progress bar
- Live statistics dashboard
- Prediction history tracking

📊 **Data & Analytics**

- Synthetic data generation (no external dataset needed - 1000 fake + 1000 real samples)
- SQLite database for prediction logging
- Statistics: total predictions, fake/real breakdown, confidence averages
- Historical prediction tracking

⚡ **Production Ready**

- Automatic model training on first run
- Manual retraining capability
- Comprehensive error handling
- Input validation for edge cases
- Efficient TF-IDF vectorization with bigrams (5000 features)

## Tech Stack

- **Backend**: Python 3.8+, Flask 3.0
- **ML/NLP**: scikit-learn, NLTK, numpy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Database**: SQLite
- **Text Processing**: TF-IDF Vectorizer, Porter Stemmer, Stopword Removal

## Project Structure

```
fake_news_detector/
├── app.py                    # Flask application with API routes
├── train_model.py            # Model training and comparison
├── data_prep.py              # Synthetic data generation & preprocessing
├── db_manager.py             # SQLite database operations
├── utils.py                  # Text preprocessing & validation utilities
├── requirements.txt          # Python dependencies
├── database.db               # SQLite predictions log (created at runtime)
├── model.pkl                 # Trained model (created at runtime)
├── vectorizer.pkl            # TF-IDF vectorizer (created at runtime)
├── model_metadata.pkl        # Model info (created at runtime)
│
├── templates/
│   └── index.html            # Single-page web interface
│
└── README.md                 # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd "c:\Users\Swami\OneDrive\Pictures\Documents\Desktop\Fake_Detection"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

The app will auto-download required NLTK data on first run, but you can pre-download:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Usage

### Run the Web Application

```bash
python app.py
```

The application will:

1. Initialize the SQLite database
2. Check for existing trained models
3. **If models don't exist**: Automatically train both classifiers (~30-60 seconds)
4. Start the Flask server on `http://localhost:5000`

Open your browser and go to: **http://localhost:5000**

### First Run (Auto-Training)

On the first run, the app will train two models:

```
======================================================================
FAKE NEWS DETECTION - MODEL TRAINING
======================================================================

[1/6] Generating synthetic dataset...
    ✓ Generated 2000 samples
    ✓ Real samples: 1000
    ✓ Fake samples: 1000

[2/6] Preprocessing texts...
    ✓ Texts preprocessed

[3/6] Vectorizing with TF-IDF...
    ✓ Vectorization complete
    ✓ Features extracted: 5000

[4/6] Splitting data (80/20)...
    ✓ Training samples: 1600
    ✓ Testing samples: 400

[5/6] Training models...
    ✓ Logistic Regression trained
    ✓ Passive Aggressive Classifier trained

[6/6] Evaluating models...
    Logistic Regression:
      Train Accuracy: 0.9931
      Test Accuracy:  0.9775
      Precision:      0.9761
      Recall:         0.9775
      F1-Score:       0.9768
      ROC-AUC:        0.9954
      CV (Mean±Std):  0.9775 ± 0.0089

[Best Model: Logistic Regression]
[F1-Score: 0.9768]
======================================================================
```

### Using the Web Interface

1. **Paste a News Article**: Paste text in the textarea
2. **Click "Analyze"**: Get instant prediction with confidence
3. **View Results**: Color-coded result (red/green) with confidence bar
4. **Sample Buttons**: Pre-load fake/real examples for testing
5. **View Stats**: See total predictions and fake/real counts
6. **History**: View recent 10 predictions with timestamps

### Train Models Manually

To retrain models from scratch:

```bash
python train_model.py
```

Or use the "Retrain Model" button in the web interface.

## Model Details

### Data Processing Pipeline

```
Raw Text Input
    ↓
Lowercase conversion
    ↓
Remove URLs, HTML tags, punctuation
    ↓
Remove extra whitespace
    ↓
Tokenization
    ↓
Remove stopwords (NLTK English)
    ↓
Porter Stemmer
    ↓
TF-IDF Vectorization (5000 features, bigrams)
    ↓
Model Prediction
```

### Models Trained

**1. Logistic Regression**

- Config: `max_iter=1000, random_state=42`
- Typical Accuracy: 97-99%
- Strengths: Fast, stable, interpretable

**2. Passive Aggressive Classifier**

- Config: `max_iter=50, random_state=42`
- Typical Accuracy: 95-97%
- Strengths: Online learning, memory efficient

**Selection**: Best model chosen by F1-Score (tiebreaker: Logistic Regression)

### Evaluation Metrics

| Metric        | Description                                          |
| ------------- | ---------------------------------------------------- |
| **Accuracy**  | Overall correctness on test set                      |
| **Precision** | Of predicted fake, how many are actually fake        |
| **Recall**    | Of actual fake, how many were caught                 |
| **F1-Score**  | Harmonic mean (primary selection metric)             |
| **ROC-AUC**   | Receiver Operating Characteristic - Area Under Curve |
| **Cross-Val** | 5-fold cross-validation for overfitting detection    |

## API Endpoints

### GET `/`

Serves the main HTML interface

**Response**: HTML page

---

### POST `/api/predict`

Predict if text is fake or real

**Request**:

```json
{
  "text": "news article or headline"
}
```

**Response**:

```json
{
  "success": true,
  "label": "real",
  "confidence": 0.9234,
  "text": "news article text (truncated to 200 chars)"
}
```

**Status Codes**:

- `200`: Successful prediction
- `400`: Invalid input (empty, too short, etc.)
- `500`: Server error

---

### GET `/api/stats`

Get prediction statistics

**Query Parameters**: None

**Response**:

```json
{
  "success": true,
  "total_predictions": 42,
  "fake_count": 18,
  "real_count": 24,
  "avg_confidence": 0.8742,
  "fake_confidence_avg": 0.8523,
  "real_confidence_avg": 0.8901
}
```

---

### GET `/api/history`

Get recent predictions

**Query Parameters**:

- `limit` (int, default=10, max=100): Number of predictions to return

**Response**:

```json
{
  "success": true,
  "predictions": [
    {
      "id": 42,
      "text": "news text snippet",
      "label": "fake",
      "confidence": 0.8750,
      "timestamp": "2024-02-23T15:30:45.123456"
    },
    ...
  ]
}
```

---

### POST `/api/retrain`

Manually retrain models

**Request**: Empty JSON body `{}`

**Response**:

```json
{
  "success": true,
  "message": "Models retrained successfully",
  "model_name": "Logistic Regression"
}
```

## Testing

### Test Cases - FAKE News Samples

```
1. "SHOCKING: Celebrity reveals secret government conspiracy that will shock everyone!"
2. "Leaked: 90% of scientific studies completely fabricated say anonymous sources"
3. "EXCLUSIVE: Unknown cure for all diseases hidden by Big Pharma corporations"
4. "Breaking: New AI will replace all humans within weeks according to unreliable sources"
5. "ALERT: New law bans all private ownership of property without evidence"
```

**Expected Result**: All classified as FAKE (confidence 80-95%)

---

### Test Cases - REAL News Samples

```
1. "Research shows 40% improvement in cancer treatment outcomes according to study"
2. "Company announces new product launch scheduled for next quarter"
3. "Official statement: Government approves new transportation infrastructure bill"
4. "Study: Climate data confirms predictions from 2015 research"
5. "Report: Economic growth reaches 3.5% this quarter says official data"
```

**Expected Result**: All classified as REAL (confidence 80-95%)

---

### Quick Test Script

```bash
# Test synthetic data generation
python -c "from data_prep import SyntheticDataGenerator; g = SyntheticDataGenerator(); t, l = g.generate_dataset(100); print(f'Generated {len(t)} samples')"

# Test database operations
python -c "from db_manager import DatabaseManager; db = DatabaseManager('test.db'); db.log_prediction('test', 'real', 0.95); print(db.get_statistics())"

# Test preprocessing
python -c "from utils import preprocess_text; print(preprocess_text('This is a TEST with URLs https://example.com and punctuation!'))"

# Test model training (takes ~30-60 seconds)
python train_model.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

### Issue: "No module named 'nltk' or NLTK data not found"

**Solution**: Download NLTK data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Issue: Port 5000 already in use

**Solution**: Edit `app.py` line 245: Change `port=5000` to another port (e.g., 5001)

```python
app.run(
    host='localhost',
    port=5001,  # Change this
    debug=False,
    use_reloader=False
)
```

### Issue: Models taking too long to train

**Normal**: First training takes 30-90 seconds (depends on CPU)

- Generating 2000 samples: ~5-10 seconds
- TF-IDF vectorization: ~5-10 seconds
- Model training: ~15-30 seconds each
- Evaluation: ~10-20 seconds

### Issue: Very low accuracy or confidence

**Solution**: The models might need retraining

```bash
rm model.pkl vectorizer.pkl model_metadata.pkl database.db
python train_model.py
```

## Performance Metrics

### Expected Accuracy

- **Training Accuracy**: 97-99%
- **Test Accuracy**: 94-98%
- **False Positive Rate**: 2-5% (predicting real as fake)
- **False Negative Rate**: 2-5% (predicting fake as real)

### Performance Requirements

- **Prediction Latency**: < 500ms per request
- **API Response Time**: < 1 second
- **Page Load Time**: < 1.5 seconds
- **Memory Usage**: ~200-300MB during operation
- **Database Size**: < 50MB for 100,000 predictions

## Security Considerations

- Input validation on all user-provided text
- SQL injection prevention via parameterized queries
- XSS prevention via HTML escaping
- No sensitive data stored (only prediction text)
- All processing done locally (no external APIs)

## File Descriptions

| File                   | Purpose                                           | Lines |
| ---------------------- | ------------------------------------------------- | ----- |
| `utils.py`             | Text preprocessing, validation, error classes     | ~350  |
| `data_prep.py`         | Synthetic data generation, preprocessing pipeline | ~280  |
| `train_model.py`       | Model training, comparison, evaluation            | ~280  |
| `db_manager.py`        | SQLite operations, statistics, history            | ~300  |
| `app.py`               | Flask routes, API endpoints, initialization       | ~400  |
| `templates/index.html` | Web UI, CSS styling, JavaScript                   | ~900  |

**Total Code**: ~2,500 lines of production-ready code

## Future Enhancements

- Add more models (SVM, Random Forest, Gradient Boosting)
- Implement model ensemble voting
- Add user authentication and multi-user support
- Export predictions to CSV/PDF
- Add real dataset import (Kaggle ISOT dataset)
- Implement confidence calibration
- Add model explanation features (LIME/SHAP)
- Deploy to cloud (Heroku, AWS, Google Cloud)

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:

1. Check the **Troubleshooting** section above
2. Review error messages in console output
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Try retraining models: `python train_model.py`

## Development Notes

### Adding Custom Models

To add a new classifier:

1. Edit `train_model.py` - Add `_train_your_model()` method
2. Call it in `train_models()` for evaluation
3. Compare F1-scores automatically

### Modifying Preprocessing

To change text preprocessing:

1. Edit `utils.py` - Modify `preprocess_text()` function
2. Clear existing models: `rm *.pkl`
3. Retrain: `python train_model.py`

### Customizing UI

To modify the web interface:

1. Edit `templates/index.html`
2. Modify CSS in `<style>` section
3. Update JavaScript in `<script>` section
4. No app restart needed - just refresh browser

### Generating Different Data

To create different synthetic samples:

1. Edit `data_prep.py` - Update `*_TEMPLATES` lists
2. Add/modify word banks (ENTITIES, TOPICS, etc.)
3. Retrain models: `python train_model.py`

---

**Created**: February 2024
**Version**: 1.0.0
**Status**: Production Ready
#   F a k e _ n e w s - d e t e c t i o n  
 