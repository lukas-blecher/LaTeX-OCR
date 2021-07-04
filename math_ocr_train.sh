echo "Creating virtual environment..."
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
echo "Virutal environment activated..."
pip3 install -r requirements.txt
pip3 install numpy wandb gdown albumentations
echo "Downloading datasets..."
gdown https://drive.google.com/uc?id=176PKaCUDWmTJdQwc-OfkO0y8t4gLsIvQ
gdown https://drive.google.com/uc?id=1QUjX6PFWPa-HBWdcY-7bA5TRVUnbyS1D
gdown https://drive.google.com/uc?id=147uMjW5br-gbmAQ42Q1i0W3TbG7xS1hl
echo "Extracting formulas..."
unzip -q formulae.zip
echo "Exraction done..."
python3 dataset/dataset.py --equations ./math.txt --images ./train --tokenizer ./tokenizer.json --out ./dataset/data/train.pkl
python3 dataset/dataset.py --equations ./math.txt --images ./test --tokenizer ./tokenizer.json --out ./dataset/data/test.pkl
python3 dataset/dataset.py --equations ./math.txt --images ./val --tokenizer ./tokenizer.json --out ./dataset/data/val.pkl
nohup python3 train.py --config ./settings/default.yaml --debug > train.log 2>&1 &
echo "Model training started as background process, check LaTeX-OCR/train.log for details..."

