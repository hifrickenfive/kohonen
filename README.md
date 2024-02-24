 # Welcome
This repo implements a kohonen map given the specifications in docs/kohonen.ipynb.  
Deviations from the specifications are permitted. 

# Requirements
pip install -r requirements.txt

# Usage 
## Terminal
python main.py  
python main.py TC4_config.yaml

## Streamlit (local)
streamlit run streamlit.py  
Then go to localhost:8501 on your web browser

## Streamlit (web)
https://kohonen.streamlit.app/

## ML Flow (local)
mlflow server --host 127.0.0.1 --port 8080    
Then go to localhost:8080 on your web browser

## Docker
docker run -p 8501:8501 <image name>  
Then go to localhost:8501 on your web browser

# Profiling
To profile main.py run the below  
python -m cProfile -o profile_output.pstats main.py  
snakeviz profile_output.pstats  