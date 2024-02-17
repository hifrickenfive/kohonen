import streamlit as st
from main import run_main_function


def on_button_click():
    st.session_state.config = {
        "grid_width": st.session_state.grid_width,
        "grid_height": st.session_state.grid_height,
        "max_iter": st.session_state.max_iter,
        "learning_rate": st.session_state.learning_rate,
        "num_input_vectors": st.session_state.num_input_vectors,
        "dim_of_input_vector": 3,
        "random_seed": st.session_state.random_seed,
    }
    st.session_state.run_id += 1


st.subheader("Kohonen Map Training App!")

# Initialize session state
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

if "config" not in st.session_state:
    st.session_state.config = dict()

if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = -1  #

# Inputs
st.session_state.grid_width = st.sidebar.slider("Grid Width", 0, 100, 10)
st.session_state.grid_height = st.sidebar.slider("Grid Height", 0, 100, 10)
st.session_state.num_input_vectors = st.sidebar.slider(
    "Number of Input Vectors", 1, 20, 20
)
st.session_state.max_iter = st.sidebar.slider("Max Iterations", 0, 1000, 500)
st.session_state.learning_rate = st.sidebar.slider("Learning Rate", 0.0, 1.0, 0.1)
st.session_state.random_seed = st.sidebar.slider("Random Seed", 0, 100, 40)


# Weclome and introduction
text_container = st.container()
welcome_text = (
    "This Kohonen Self Organizing Map (SOM) provides as visualisation technique to help "
    "understand high dimensional data by reducing the dimensions of data to a map. "
    "It is an unsupervised learning technique. "
    "In this example, we are mapping a number of input vectors, each with dimension 3, onto a 2D grid. Select the parameters in the sidebar and click submit."
)
with text_container:
    st.markdown(
        f'<div style="white-space: pre-line;">{welcome_text}</div>',
        unsafe_allow_html=True,
    )  # unsafe_allow_html is used to allow line breaks in the text

if st.button("Submit"):
    on_button_click()

if (
    st.session_state.run_id != 0
    and st.session_state.run_id > st.session_state.last_run_id
):
    with st.spinner("Training..."):
        (
            st.session_state.fig_input,
            st.session_state.fig_initial_grid,
            st.session_state.fig_trained_grid,
            st.session_state.log,
        ) = run_main_function(st.session_state.config)

    st.session_state.last_run_id = st.session_state.run_id

if "fig_input" in st.session_state:
    # st.markdown("---")
    st.text("Input vector represented as pixels")
    st.pyplot(st.session_state.fig_input)
    # st.markdown("---")

middle_col, right_col = st.columns(2)

with middle_col:
    if "fig_initial_grid" in st.session_state:
        st.text("Map Before: Randomly Initialised 2D-Grid")
        st.pyplot(st.session_state.fig_initial_grid)

with right_col:
    if "fig_trained_grid" in st.session_state:
        st.text("Map After: After Training")
        st.pyplot(st.session_state.fig_trained_grid)

if "log" in st.session_state:
    st.write(f"Elapsed time: {st.session_state.log['Elapsed time']:.3f} seconds")
