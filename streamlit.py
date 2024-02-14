import streamlit as st
from main import run_main_function
import matplotlib.pyplot as plt


def create_plt_placeholder():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_facecolor("#f0f0f0")  # Set background color to neutral gray
    ax.axis("off")  # Turn off axes
    plt.xlim(0, 640)  # Set x-axis limits
    plt.ylim(0, 480)  # Set y-axis limits
    return fig


st.subheader("Kohonen Map Training App")

# Get user inputs
st.sidebar.subheader("Enter parameters")
grid_width = st.sidebar.slider("Grid Width", min_value=0, max_value=100, value=10)
grid_height = st.sidebar.slider("Grid Height", min_value=0, max_value=100, value=10)
num_input_vectors = st.sidebar.slider(
    "Number of Input Vectors", min_value=1, max_value=20, value=20
)
dim_of_input_vector = st.sidebar.slider(
    "Dimension of Input Vector", min_value=1, max_value=3, value=3
)
max_iter = st.sidebar.slider("Max Iterations", min_value=0, max_value=1000, value=500)
learning_rate = st.sidebar.slider(
    "Learning Rate", min_value=0.0, max_value=1.0, value=0.1
)
random_seed = st.sidebar.slider("Random Seed", min_value=0, max_value=100, value=40)


text_container = st.container()
long_text = "The Kohonen Self Organizing Map (SOM) provides a data visualization technique which helps to understand high dimensional data by reducing the dimensions of data to a map. SOM also represents clustering concept by grouping similar data together. Unlike other learning technique in neural networks, training a SOM requires no target vector. A SOM learns to classify the training data without any external supervision."
with text_container:
    st.markdown(
        f'<div style="white-space: pre-line;">{long_text}</div>',
        unsafe_allow_html=True,  # allow user to modify the page
    )

st.markdown("---")

middle_col, right_col = st.columns(2)
plt_placeholder = create_plt_placeholder()


with middle_col:
    st.text("Before: Randomly Initialised 2D-Grid")
    plot_placeholder_before = st.empty()

with right_col:
    st.text("After: Map of Input Vectors to 2D-Grid")
    plot_placeholder_after = st.empty()


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

    with st.spinner("Training..."):
        fig_initial_grid, fig_trained_grid = run_main_function(config_dict)

    with right_col:
        plot_placeholder_before.pyplot(fig_initial_grid)
        plot_placeholder_after.pyplot(fig_trained_grid)
