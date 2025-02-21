from sqlalchemy import Column, Integer, String, Float, DateTime
from . import app
import datetime


# Table 1: EEG_Data Table (Raw and processed EEG data with labels)
class EEGData(app.db.Model):
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, nullable=False)
    session_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    raw_data = Column(String)  # Raw EEG signal (can be serialized data)
    processed_data = Column(String)  # Processed features (e.g., frequency bands)
    label = Column(Integer, nullable=False)  # Label for classification


# Table 2: Model_Training Table (Model training parameters and results)
class ModelTraining(app.db.Model):
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    epochs = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)  # Model accuracy after training
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# Table 3: Image_Output Table (Storing images like confusion matrices or charts)
class ImageOutput(app.db.Model):
    id = Column(Integer, primary_key=True)
    training_id = Column(Integer, nullable=False)  # Linking to a model training session
    image_type = Column(String, nullable=False)  # e.g., 'confusion_matrix'
    image_data = Column(String, nullable=False)  # Base64-encoded image or file path
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


