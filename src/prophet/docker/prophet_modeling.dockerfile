FROM continuumio/miniconda3

WORKDIR /app

# Create the environment by adding layers:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]

# Following CMD keeps the container running. CMD runs the default 
CMD ["tail", "-f", "/dev/null", "&", "jupyter", "lab", "--port=8888", "--no-browser"]

# Doesn't directly expose port except for inter-container comm
EXPOSE 8888
