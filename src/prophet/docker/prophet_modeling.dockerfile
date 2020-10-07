FROM continuumio/miniconda3

WORKDIR /app

# Create the environment by adding layers:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# CMD runs the default
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "-allow-root", '--notebook-dir="/"']

# Doesn't directly expose port except for inter-container comm
EXPOSE 8888


