# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install pytest for running tests
RUN pip install pytest

# Run unit tests
RUN pytest tests/

# Set the default command to run the optimization example
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]
