# 1. Use an official lightweight Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project into the container
COPY . .

# 6. Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# 7. Create a shell script to run both FastAPI and Streamlit
RUN echo '#!/bin/bash\n\
uvicorn app:app --host 0.0.0.0 --port 8000 &\n\
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh

RUN chmod +x start.sh

# 8. Run the script
CMD ["./start.sh"]