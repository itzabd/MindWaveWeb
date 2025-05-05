# Use an official Python runtime as the base image


FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask-mail
RUN pip install --no-cache-dir matplotlib pandas
RUN pip install --no-cache-dir cloudinary

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the Flask application
#CMD ["python", "app.py"]
CMD ["flask", "run", "--host=0.0.0.0"]