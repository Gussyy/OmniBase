# omnibase as a service. Mount your datasets and point the UI at their paths inside the container.
#   docker build -t omnibase . && docker run -p 8000:8000 -v /data:/data omnibase
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY omnibase ./omnibase
RUN pip install --no-cache-dir ".[data,service]"
ENV OMNIBASE_WORK=/work
VOLUME ["/work", "/data"]
EXPOSE 8000
CMD ["omnibase", "serve", "--host", "0.0.0.0", "--port", "8000"]
