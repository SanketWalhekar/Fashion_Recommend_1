import streamlit as st
from streamlit.components.v1 import html

def welcome_page():
    st.title('Fashion Recommendation System')
    st.write("""
    Welcome to the Fashion Recommender System! 
    Upload an image to find similar fashion items.
    """)
    if st.button("Find Recommendations", key='recommendation_button'):
        # Redirect to the second page
        html_code = """
        <script>
            window.location.href = "example.py";
        </script>
        """
        st.components.v1.html(html_code)

def recommendation_page():
    pass
    # This function remains unchanged


def main():
    if "page" not in st.session_state:
        st.session_state.page = "Welcome"

    if st.session_state.page == "Welcome":
        welcome_page()
    elif st.session_state.page == "example":
        recommendation_page()
    # This function remains unchanged

if __name__ == "__main__":
    main()
