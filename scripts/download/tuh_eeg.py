import subprocess


cmd = 'sshpass -p {pass} rsync --list-only nedc-eeg@www.isip.piconepress.com:data/eeg/tuh_eeg/v2.0.0/edf/002/'

res = subprocess.run(cmd.split(), capture_output=True).stdout
files = res.decode().split('\n')

# files = list(filter(lambda x: x.endswith('.edf'), files))

print(len(files))
print()
print('\n'.join(files[:10]))
