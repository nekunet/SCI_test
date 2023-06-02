import streamlit as st


def run():
    st.set_page_config(
        page_title="Acknowledgement",
        page_icon="🙏",
    )

st.write("# Acknowledgement")

    
st.markdown(
        """
        We acknowledge the Japan Association of Rehabilitation Database for establishing the Japan Rehabilitation Database, which served as a vital resource for this study. The content and predictions of this web application reflects views of the authors, not the views of the Japanese Association of Rehabilitation Database. The registration data is not a representative sample of rehabilitation in Japan as well as rehabilitation in the user's country. Most of the facilities participated are actively engaged in the rehabilitation of spinal cord injuries, which could be a bias in the prediction model.

    """
    )
