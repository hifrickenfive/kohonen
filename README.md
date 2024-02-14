# Welcome
This repo implements a kohonen map given the specifications in docs/kohonen.ipynb.
Deviations from the specifications are permitted. 

# Requirements
pip install -r requirements.txt

# Usage (terminal)
python main.py 
python main.py <path_to_config_file> <plot_option>

# Usage (streamlit)
streamlit run streamlit.py

## Profiling
To profile main.py run the below
python -m cProfile -o profile_output.pstats main.py
snakeviz profile_output.pstats

# Licene
MIT License