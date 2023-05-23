touch ~/.no_auto_tmux
apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt install python3.7 -y
apt install virtualenv -y
apt install unzip -y

virtualenv --python="python3.7" ".venv"
