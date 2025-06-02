FROM python:3.11-slim
# set the working directory
WORKDIR /app
# copy the requirements file
COPY requirements.txt .
# create the virtual environment in the app directory
ENV VIRTUAL_ENV_PATH=/app/.venv
RUN python3 -m venv venv
# set the PATH to include the virtual environment
ENV PATH="$VIRTUAL_ENV_PATH/bin:$PATH"
# install the requirements
RUN pip install --no-cache-dir -r requirements.txt
# copy the rest of the application code
COPY . .
# expose the uvicorn port
EXPOSE 8000
CMD ["fastapi", "run", "src/main.py"]
#CMD ["uvicorn", "src/main:app", "--host", "0.0.0.0", "--port", "8000"]