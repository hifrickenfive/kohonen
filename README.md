# Welcome
This repo implements a kohonen map given the specifications in docs/kohonen.ipynb.
Deviations from the specifications are permitted. 

# Requirements
pip install -r requirements.txt

# Usage (terminal)
python main.py 
python main.py <config_file_name>

# Usage (streamlit)
streamlit run streamlit.py

# Usage (docker)
docker run -p 8501:8501 kohonen-st-app

## Profiling
To profile main.py run the below
python -m cProfile -o profile_output.pstats main.py
snakeviz profile_output.pstats

# Licene
MIT License