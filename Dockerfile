FROM python:3.11-slim

# Create a non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy all project files into the container, changing ownership to the non-root user
COPY --chown=user:user . /app/

# Install project and dependencies
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

# Make the start script executable
RUN chmod +x /app/start.sh

# Expose both ports (7860 for Gradio, 8000 for FastAPI)
EXPOSE 7860
EXPOSE 8000

# Use the startup script to launch both processes
CMD ["/app/start.sh"]
