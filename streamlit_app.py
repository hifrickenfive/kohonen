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
        "radius_decay_factor": st.session_state.radius_decay_factor,
        "influence_decay_factor": st.session_state.influence_decay_factor,
    }
    st.session_state.run_id += 1


st.subheader("Kohonen Map Training App!")

tab1, tab2 = st.tabs(["Generate Map", "How does it work?"])

with tab1:
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
    st.session_state.radius_decay_factor = st.sidebar.slider(
        "Radius Decay Factor", 0.1, 1.0, 0.2
    )
    st.session_state.influence_decay_factor = st.sidebar.slider(
        "Influence Decay Factor", 0.1, 10.0, 1.0
    )

    # Weclome and introduction
    text_container = st.container()
    welcome_text = (
        "This Kohonen Self Organizing Map (SOM) provides a visualisation technique to help "
        "understand high dimensional data by reducing the dimensions of data to a map. "
        "It is an unsupervised learning technique. "
        "In this example, we are mapping a number of input vectors, each with dimension 3, onto a 2D grid. Select the parameters in the sidebar and click submit."
    )
    with text_container:
        st.markdown(
            f'<div style="white-space: pre-line;">{welcome_text}</div>',
            unsafe_allow_html=True,
        )  # unsafe_allow_html is used to allow line breaks in the text

    if st.button("Generate Map"):
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
        st.write(f"Elapsed time: {st.session_state.log['elapsed_time']:.3f} seconds")

with tab2:
    # left_col_tab2, right_col_tab2 = st.columns(2)
    # with left_col_tab2:

    text_container = st.container()
    explanatory_text = (
        "1. Initialise all the nodes in the grid.\n"
        "2. Enumerate through the training data (repeating if necessary) to select the currrent input vector. \n"
        "3. Find the node in the grid with the weights most similar to the input vector. This is the Best Matching Unit (BMU). \n"
        "4. Calculate the radius of the neighbourhood around the BMU. The radius starts large and decays with each time-step.\n"
        "5. Adjust the weights of all the nodes inside the radius to be more similar to the input vector. The closer a node is to the BMU, the more its weights get altered. \n"
        "6. Return to step 2 until we've completed N iterations. \n"
    )
    st.markdown(
        f'<div style="white-space: pre-line; font-size: 14px;">{explanatory_text}</div>',
        unsafe_allow_html=True,
    )

    # with right_col_tab2:
    video_file = open("videos\\ani_after_bug_fixes.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
