# Step 1: Use an official Python base image.
# We're using a specific version for consistency.
FROM python:3.9-slim

# Step 2: Set the working directory inside the container.
# This is where our app's files will live.
WORKDIR /app

# Step 3: Copy the requirements file into the container.
COPY requirements.txt .

# Step 4: Install the Python libraries listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application code into the container.
COPY . .

# Step 6: Expose port 8000.
# This tells Docker that the container will listen for traffic on this port.
EXPOSE 8000

# Step 7: Define the command to run your application.
# This starts the Uvicorn server, making it accessible from outside.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
