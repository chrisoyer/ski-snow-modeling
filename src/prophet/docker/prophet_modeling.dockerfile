FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# The code to run when container is started:
#COPY run.py .
ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]

# Following CMD keeps the container running
# Modify CMD to run the app that you require. 
CMD tail -f /dev/null &
CMD ["jupyter", "lab", "--port=8888", "--no-browser"]

EXPOSE 8888
