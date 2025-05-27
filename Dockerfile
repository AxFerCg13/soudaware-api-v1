FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# 
ENV PIP_ROOT_USER_ACTION=ignore

# Clonar el repositorio
RUN git clone https://github.com/AxFerCg13/soudaware_api_v1.git .
WORKDIR /app
COPY soundaware.json ./services/
RUN git fetch --all && git pull
# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install fastapi uvicorn scipy && \
    pip install -r requirements.txt

EXPOSE 8000

# Iniciar contenedor
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]