#SETUP
pip install -r requirements.txt

python3 -m venv venv

brew install libmagic poppler tesseract libreoffice pandoc #Pdf isleme icin gerekli


Windows: .\venv\Scripts\activate

MacOS: source venv/bin/activate

deactivate

#youtube transkriptlerini almak icin bu gerekti, belirtmek gerekebilir
brew install ffmpeg


# IMPORTANT NOTE: The version v.0.1.21 of RAGAS has been used to create the test data. 
# The higher version v.0.2x has a significantly different test data generation structure, which is why v.0.1.21 was preferred.
# The existing test data CSV files are already located in the corresponding data folder.
# If you want to create new test data with this notebook, please use a different environment and install it with `pip install ragas==0.1.21`.
# The latest version has been used for evaluation with RAGAS metrics, 
# and this version is specified in the requirements.txt file: `pip install git+https://github.com/explodinggradients/ragas.git`