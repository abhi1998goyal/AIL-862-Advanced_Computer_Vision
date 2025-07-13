import subprocess
import sys
import os
import zipfile

def unzip_data():
    ZIP_NAME = 'data.zip'
    if os.path.exists(ZIP_NAME):
        print(f'=== Unzipping {ZIP_NAME} ===')
        with zipfile.ZipFile(ZIP_NAME, 'r') as zf:
            zf.extractall('.')  
        print('=== Unzip complete ===')
    else:
        print(f'No zip file named {ZIP_NAME} found, skipping unzip.')
        print(f'No zip file named {{ZIP_NAME}} found, skipping unzip.')


def run_prepare_data():
    os.makedirs('processed_data', exist_ok=True)   
    ret = subprocess.run(
        [sys.executable, 'prepare_data.py'],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    if ret.returncode != 0:
        print('prepare_data.py failed with exit code', ret.returncode, file=sys.stderr)
        sys.exit(ret.returncode)


def run_train():
    os.makedirs('saved_model', exist_ok=True) 
    ret = subprocess.run(
        [sys.executable, 'train.py'],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    if ret.returncode != 0:
        print('train.py failed with exit code', ret.returncode, file=sys.stderr)
        sys.exit(ret.returncode)


def main():
    print('=== Data unzipped ===')
    unzip_data()
    print('=== Running data preparation ===')
    run_prepare_data()
    print('=== Data preparation complete ===')

    print('=== Starting training ===')
    run_train()
    print('=== Training complete ===')


if __name__ == '__main__':
    main()
