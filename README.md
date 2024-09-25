#SETUP
pip install -r requirements.txt

python3 -m venv .venv

brew install libmagic poppler tesseract libreoffice pandoc #Pdf isleme icin gerekli


Windows: .\venv\Scripts\activate

MacOS: source venv/bin/activate

deactivate

#youtube transkriptlerini almak icin bu gerekti, belirtmek gerekebilir
brew install ffmpeg
