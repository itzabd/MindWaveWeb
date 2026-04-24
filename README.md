# MindWaveWeb 🚀🧠

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Latest-000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

A sophisticated web application that leverages **Large Language Models (LLMs)** and **Machine Learning** to analyze EEG (Electroencephalography) data, providing actionable insights into cognitive patterns and brain activity.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 📖 Overview

MindWaveWeb combines cutting-edge LLM-powered classification techniques with traditional machine learning to process and interpret EEG data. The application provides researchers, clinicians, and developers with a comprehensive platform to:

- **Analyze EEG signals** with advanced classification algorithms
- **Identify cognitive patterns** and brain activity anomalies
- **Generate AI-powered insights** using language models
- **Visualize data** through interactive web-based dashboards
- **Manage user accounts** with secure authentication

The system is built on a modular architecture that separates data processing (Python/Jupyter), model training (scikit-learn, PyTorch), and web presentation (Flask, PostgreSQL) layers.

## 🎯 Key Features

### EEG Data Processing
- **Multi-channel EEG analysis** (9-channel support)
- **Data normalization and preprocessing** using StandardScaler
- **Train/validation/test split** automation with configurable ratios
- **Support for large datasets** (360,000+ row datasets)

### Machine Learning Classification
- **MLP (Multi-Layer Perceptron)** neural networks with hyperparameter optimization
- **Grid search** for optimal model parameters
- **Pipeline architecture** for streamlined preprocessing and prediction
- **Multi-class classification** support with ROC-AUC metrics
- **Fine-tuned DistilGPT2** model for LLM-powered analysis

### Web Application
- **User authentication** with secure login/registration
- **Flask-based REST API** for data management
- **Interactive dashboards** for EEG visualization
- **Cloudinary integration** for file management
- **Supabase backend** for scalable database operations
- **Session management** with configurable timeouts
- **File upload support** (10 MB limit)

### Data Insights
- **Real-time classification** on uploaded EEG data
- **Automated report generation** with classification metrics
- **Visual analytics** including accuracy, ROC-AUC, and classification reports
- **Historical data tracking** for longitudinal analysis

## 🛠 Technology Stack

### Backend
| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Flask** | Web framework and REST API |
| **scikit-learn** | Machine learning algorithms |
| **pandas** | Data manipulation and analysis |
| **joblib** | Model serialization |
| **Flask-Login** | User authentication |
| **Supabase** | PostgreSQL database backend |
| **python-dotenv** | Environment configuration |

### Frontend & Data Visualization
| Technology | Purpose |
|---|---|
| **HTML5** | Markup structure |
| **CSS3** | Styling and responsive design |
| **Jinja2** | Template rendering |
| **Matplotlib** | Data visualization |
| **Cloudinary** | Cloud-based file management |

### Development & ML
| Technology | Purpose |
|---|---|
| **Jupyter Notebooks** | Interactive data exploration |
| **PyTorch/Transformers** | Deep learning & NLP models |
| **PEFT** | Parameter-efficient fine-tuning |

## 🏗 Architecture

```
MindWaveWeb/
├── Data Processing Layer
│   ├── data_split.py          # Dataset splitting utilities
│   ├── jupyter_notebooks/     # EDA and model exploration
│   └── models/                # Trained model artifacts
│
├── ML Classification Layer
│   ├── mlp_eeg_train.py       # MLP model training pipeline
│   ├── mlp_eeg_test.py        # Model evaluation and testing
│   └── best_model.joblib      # Serialized trained model
│
├── Web Application Layer
│   ├── app.py                 # Flask application entry point
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JavaScript assets
│   └── requirements.txt        # Python dependencies
│
└── Configuration
    └── .env                   # Environment variables
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database (or Supabase account)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/itzabd/MindWaveWeb.git
cd MindWaveWeb
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root directory with the following configuration:

```env
# Database Configuration (PostgreSQL)
PG_USER=your_database_username
PG_PASS=your_database_password
PG_HOST=your_database_host
PG_PORT=5432

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Cloudinary Configuration (for file uploads)
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# Email Configuration (optional)
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

### Database Setup

1. Create a PostgreSQL database
2. Update `.env` with your database credentials
3. Run migrations (if applicable)

## 🚀 Usage

### Running the Web Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Training an EEG Classification Model

```bash
python mlp_eeg_train.py
```

**Configuration in `mlp_eeg_train.py`:**
```python
input_csv = 'data/test_mlp.csv'          # Your input EEG data
output_model = 'best_model.joblib'       # Output model path
```

### Testing a Trained Model

```bash
python mlp_eeg_test.py
```

**Configuration in `mlp_eeg_test.py`:**
```python
test_csv_path = 'data/test_mlp.csv'      # Test dataset path
model_file_path = 'models/best_model.joblib'  # Trained model
```

### Data Preparation

```bash
python data_split.py
```

Automatically splits your EEG dataset into:
- **60%** Training data
- **20%** Validation data
- **20%** Test data

## 🔧 Development

### Project Structure Details

**Data Processing:**
- EEG data in CSV format (channels as columns, samples as rows)
- Automatic normalization using StandardScaler
- Configurable data splitting ratios

**Model Training:**
- Hyperparameter optimization via GridSearchCV
- Cross-validation for robust performance estimation
- Multiple architectures tested (hidden layer sizes, activation functions)
- Evaluation metrics: Accuracy, ROC-AUC, Precision, Recall, F1-Score

**Web Interface:**
- User authentication with password hashing
- Session-based access control (30-minute timeout)
- File upload with security constraints
- Real-time model inference

### Extending the Application

1. **Add new ML models:** Implement in Python, serialize with joblib
2. **Extend web features:** Add routes in `app.py`, templates in `templates/`
3. **Enhance data processing:** Add notebooks to `jupyter_notebooks/`

## 📊 Expected Input/Output

### Input Format (CSV)
```
Channel_1, Channel_2, ..., Channel_9, label
-0.123,    0.456,    ..., -0.789,     0
0.234,     -0.567,   ..., 0.890,      1
...
```

### Output (Classification Report)
```
precision    recall  f1-score   support

    0       0.92      0.89      0.90      150
    1       0.87      0.91      0.89      145

accuracy                         0.90      295
macro avg   0.90      0.90      0.90      295
```

## 🔐 Security Considerations

- Passwords hashed using Werkzeug security
- Environment variables for sensitive data
- Session expiration (30 minutes)
- File upload size limits (10 MB)
- CSRF protection enabled
- Secure database credentials via Supabase

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support & Contact

For questions, issues, or suggestions:
- **GitHub Issues:** [Create an issue](https://github.com/itzabd/MindWaveWeb/issues)
- **Author:** [@itzabd](https://github.com/itzabd)

## 🙏 Acknowledgments

- EEG data analysis methodologies from neuroscience research
- scikit-learn machine learning library
- Flask web framework community
- Supabase for backend infrastructure
- Cloudinary for file management services

---

**Last Updated:** April 24, 2026  
**Status:** Active Development
