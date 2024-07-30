# Use the base image
FROM pytorchlightning/pytorch_lightning:latest

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install required Python packages
RUN pip install \
    datasets==2.19.1 \
    transformers==4.42.3 \
    timm==1.0.7 \
    spacy==3.7.5 \
    beautifulsoup4==4.12.3 \
    accelerate==0.25.0 \
    easyocr==1.7.1 \
    bert-score==0.3.13 \
    pyemd==1.00 \
    ipython==8.26.0 \
    git+https://github.com/KennyNg-19/emnlp19-moverscore.git

# Download the spaCy model
RUN python -m spacy download en_core_web_sm

# Ensure the main script is executable
RUN chmod +x main_script.sh
