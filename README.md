## Advancing IT Support Services with Generative AI: Leveraging Large Language Models and Retrieval Augmented Generation for Domain-Specific Chatbots

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Virtual Environment
#### For MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```
#### For Windows
```bash
python -m venv venv
.\venv\Scripts\activate

deactivate
```
```bash
### Create .env file, and retrieve API keys from 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
```

### Run Streamlit Chatbot (example)
```bash
streamlit run app_fusion.py
streamlit run graph_fusion.py 
```
### Testset and Data

#### [Drive Link](https://drive.google.com/drive/folders/15oI_Kaivu39pkNFeSbLzGO3kA0HhBOBG?usp=sharing)

#### IMPORTANT NOTE:
> The version v.0.1.21 of RAGAS has been used to create the test data. 
The higher version v.0.2x has a significantly different test data generation structure, which is why v.0.1.21 was preferred.
The existing test data CSV files are already located in the corresponding data folder.
If you want to create new test data with this notebook, please use a different environment and install it with `pip install ragas==0.1.21`.
The latest version has been used for evaluation with RAGAS metrics, 
and this version is specified in the requirements.txt file: `pip install git+https://github.com/explodinggradients/ragas.git`