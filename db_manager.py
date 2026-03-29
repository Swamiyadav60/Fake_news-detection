"""
Database management module for Fake News Detection.

This module handles SQLite database operations for logging predictions,
retrieving statistics, and managing prediction history.
"""

import sqlite3
from datetime import datetime
import json
from utils import DatabaseError


class DatabaseManager:
    """
    Manage SQLite database for prediction logging and statistics.
    """

    def __init__(self, db_path='database.db'):
        """
        Initialize database manager.

        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database with required tables and schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_text TEXT NOT NULL,
                        predicted_label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create index on timestamp for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON predictions(timestamp)
                ''')

                conn.commit()

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    def log_prediction(self, text, label, confidence):
        """
        Log a prediction to the database.

        Args:
            text (str): Original input text
            label (str): Predicted label ('fake' or 'real')
            confidence (float): Confidence score (0.0-1.0)

        Returns:
            int: ID of the inserted record

        Raises:
            DatabaseError: If logging fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO predictions (input_text, predicted_label, confidence, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (text, label, confidence, datetime.now().isoformat()))

                conn.commit()
                return cursor.lastrowid

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to log prediction: {str(e)}")

    def get_statistics(self):
        """
        Get aggregate statistics of all predictions.

        Returns:
            dict: Statistics including totals, counts, and averages
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total count
                cursor.execute('SELECT COUNT(*) FROM predictions')
                total = cursor.fetchone()[0]

                if total == 0:
                    return {
                        'total_predictions': 0,
                        'fake_count': 0,
                        'real_count': 0,
                        'avg_confidence': 0.0,
                        'fake_confidence_avg': 0.0,
                        'real_confidence_avg': 0.0
                    }

                # Get fake count
                cursor.execute(
                    "SELECT COUNT(*) FROM predictions WHERE predicted_label = 'fake'"
                )
                fake_count = cursor.fetchone()[0]

                # Get real count
                cursor.execute(
                    "SELECT COUNT(*) FROM predictions WHERE predicted_label = 'real'"
                )
                real_count = cursor.fetchone()[0]

                # Get average confidence overall
                cursor.execute('SELECT AVG(confidence) FROM predictions')
                avg_confidence = cursor.fetchone()[0] or 0.0

                # Get average confidence for fake
                cursor.execute(
                    "SELECT AVG(confidence) FROM predictions WHERE predicted_label = 'fake'"
                )
                fake_confidence_avg = cursor.fetchone()[0] or 0.0

                # Get average confidence for real
                cursor.execute(
                    "SELECT AVG(confidence) FROM predictions WHERE predicted_label = 'real'"
                )
                real_confidence_avg = cursor.fetchone()[0] or 0.0

                return {
                    'total_predictions': total,
                    'fake_count': fake_count,
                    'real_count': real_count,
                    'avg_confidence': round(avg_confidence, 4),
                    'fake_confidence_avg': round(fake_confidence_avg, 4),
                    'real_confidence_avg': round(real_confidence_avg, 4)
                }

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get statistics: {str(e)}")

    def get_recent_predictions(self, limit=10):
        """
        Get recent predictions from the database.

        Args:
            limit (int): Number of recent predictions to retrieve

        Returns:
            list: List of prediction records, newest first
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT id, input_text, predicted_label, confidence, timestamp
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))

                rows = cursor.fetchall()
                predictions = []

                for row in rows:
                    predictions.append({
                        'id': row['id'],
                        'text': row['input_text'],
                        'label': row['predicted_label'],
                        'confidence': round(row['confidence'], 4),
                        'timestamp': row['timestamp']
                    })

                return predictions

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get recent predictions: {str(e)}")

    def get_prediction_by_id(self, prediction_id):
        """
        Get a specific prediction by ID.

        Args:
            prediction_id (int): ID of the prediction

        Returns:
            dict: Prediction record or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT id, input_text, predicted_label, confidence, timestamp
                    FROM predictions
                    WHERE id = ?
                ''', (prediction_id,))

                row = cursor.fetchone()

                if not row:
                    return None

                return {
                    'id': row['id'],
                    'text': row['input_text'],
                    'label': row['predicted_label'],
                    'confidence': round(row['confidence'], 4),
                    'timestamp': row['timestamp']
                }

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get prediction: {str(e)}")

    def cleanup_old_records(self, days=30):
        """
        Delete predictions older than specified days.

        Args:
            days (int): Number of days to keep

        Returns:
            int: Number of records deleted
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    DELETE FROM predictions
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))

                conn.commit()
                return cursor.rowcount

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to cleanup old records: {str(e)}")

    def export_to_json(self, output_file='predictions_export.json', limit=None):
        """
        Export predictions to JSON file.

        Args:
            output_file (str): Output JSON file path
            limit (int): Maximum records to export (None = all)

        Returns:
            int: Number of records exported
        """
        try:
            query = 'SELECT * FROM predictions ORDER BY timestamp DESC'
            params = []

            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

                data = []
                for row in rows:
                    data.append({
                        'id': row['id'],
                        'text': row['input_text'],
                        'label': row['predicted_label'],
                        'confidence': row['confidence'],
                        'timestamp': row['timestamp']
                    })

                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)

                return len(data)

        except (sqlite3.Error, IOError) as e:
            raise DatabaseError(f"Failed to export predictions: {str(e)}")

    def get_total_count(self):
        """
        Get total number of predictions.

        Returns:
            int: Total prediction count
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM predictions')
                return cursor.fetchone()[0]

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get total count: {str(e)}")


if __name__ == "__main__":
    # Test database operations
    db = DatabaseManager('test_database.db')

    # Log some test predictions
    print("Logging test predictions...")
    db.log_prediction("Breaking news about politics", "real", 0.95)
    db.log_prediction("SHOCKING: Celebrities revealed as aliens", "fake", 0.87)
    db.log_prediction("Research shows health improvement", "real", 0.92)

    # Get statistics
    print("\nDatabase Statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get recent predictions
    print("\nRecent Predictions:")
    recent = db.get_recent_predictions(limit=5)
    for pred in recent:
        print(f"  ID {pred['id']}: {pred['label']} ({pred['confidence']}) - {pred['timestamp']}")
