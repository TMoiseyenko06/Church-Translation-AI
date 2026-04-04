FROM python:3.11-slim

WORKDIR /app

# Install cloudflared and curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
         -o /usr/local/bin/cloudflared && \
    chmod +x /usr/local/bin/cloudflared && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py booth.html listen.html start.sh ./
RUN chmod +x start.sh

# Port the relay listens on internally
EXPOSE 8000

# VAST_WS_URL must be set at runtime, e.g:
#   docker run -e VAST_WS_URL=ws://1.2.3.4:33816/ws/worker ...
ENV VAST_WS_URL=""

CMD ["./start.sh"]
