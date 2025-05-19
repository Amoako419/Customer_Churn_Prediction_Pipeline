# Use the base Astro Runtime image so we can set up our CA cert before installing any packages
FROM astrocrpublic.azurecr.io/runtime:3.0-2

# Switch to root for setup
USER root

# Install Python dependencies
COPY requirements.txt .
RUN /usr/local/bin/install-python-dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8078 for the API server
EXPOSE 8088
