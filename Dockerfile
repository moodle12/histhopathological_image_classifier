# # Use the official Python image as the base image
# FROM python:3.10-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the contents of the local "ui" directory to the working directory in the container
# COPY UI/ .

# # Install the required Python packages
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose port 80
# EXPOSE 80

# # Command to run the FastAPI application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]




FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./UI /code/UI

EXPOSE 8000

CMD ["uvicorn", "UI.main:UI", "--host", "0.0.0.0", "--port", "8000"]
