import streamlit as st
from main import run_main_function

st.title("Kohonen Map Training App")

st.subheader("Parameters")

# User inputs
grid_width = st.slider("Grid Width", min_value=0, max_value=100, value=10)
grid_height = st.slider("Grid Height", min_value=0, max_value=100, value=10)
num_input_vectors = st.slider(
    "Number of Input Vectors", min_value=1, max_value=20, value=20
)
dim_of_input_vector = st.slider(
    "Dimension of Input Vector", min_value=1, max_value=3, value=3
)
max_iter = st.slider("Max Iterations", min_value=0, max_value=1000, value=500)
learning_rate = st.slider("Learning Rate", min_value=0.0, max_value=1.0, value=0.1)
random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=40)

if st.button("Submit"):
    config_dict = {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "num_input_vectors": num_input_vectors,
        "dim_of_input_vector": dim_of_input_vector,
        "random_seed": random_seed,
    }

    fig_initial_grid, fig_trained_grid = run_main_function(config_dict)
    st.pyplot(fig_initial_grid)
    st.pyplot(fig_trained_grid)
