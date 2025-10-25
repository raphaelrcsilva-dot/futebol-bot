# Use Python slim
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
# Expose port for Render
EXPOSE 10000
# Default command: run uvicorn (Render will override start command)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]