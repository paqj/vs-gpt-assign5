import streamlit as st

st.title("Input Your OpenAI API Key on Side Bar")

with st.sidebar:
    # st.title("Input Your OpenAI API Key on Side Bar")
    # 사용자로부터 OpenAI API 키 입력 받기
    user_api_key = st.text_input("Enter your OpenAI API key", "")

    st.markdown(
        """
        [GitHub Repository](https://github.com/paqj/vs-gpt-assign5)
        """,
    )

# 입력된 API 키를 세션 상태에 저장
if user_api_key:
    st.session_state['api_key'] = user_api_key
    st.success("API Key saved successfully!")