pip3 install -r requirements.txt
mkdir weights
mkdir .data
mkdir third_party

pip3 install git+https://github.com/SPOClab-ca/dn3@bendr

apt-get install libpython-dev g++ python3.7-dev -y
cd third_party && \
    git clone https://github.com/facebookresearch/fairseq && \
    cd fairseq && pip3 install -e .
