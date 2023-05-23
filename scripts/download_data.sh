mkdir -p ../.data

# MMIDB dataset

mkdir -p ../.data/mmidb
cd ../.data/mmidb \
    && wget https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip \
    && unzip -q eeg-motor-movementimagery-dataset-1.0.0.zip
