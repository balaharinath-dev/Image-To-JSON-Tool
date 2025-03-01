import streamlit as st

st.set_page_config(page_title='OCR / HWR -> JSON',page_icon='📄')

st.title('Medical Form Images -> JSON 📄')

uploaded_file = st.file_uploader('Upload your Image', type=['jpg', 'jpeg', 'png'])

language = st.text_input('Enter the language to translate')

if uploaded_file and st.button('Convert'):

    import base64

    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=OPENAI_API_KEY)

    from langchain_core.output_parsers import JsonOutputParser

    parser = JsonOutputParser()

    from langchain_core.prompts import ChatPromptTemplate
    from langchain.schema import SystemMessage, HumanMessage

    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=[
                    {
                        'type': 'text',
                        'text': 'You are an expert AI who extracts data from the provided jpg of medical form and then construct and return a json object in a most suitable way and also translates the json content into the mentioned language!'
                    }
                ]
            ),
            HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': f'Extract the printed as well as handwritten contents from the following image and return a translated suitable json object in {language}'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}'
                        }
                    }
                ]
            )
        ]
    )

    chain = template | llm | parser

    response = chain.invoke({'base64_image': base64_image,'language':language})

    col1, col2 = st.columns([1,1])

    from PIL import Image

    with col1:
        st.text("Uploaded Form Image")
        image = Image.open(uploaded_file)
        st.image(image)

    with col2:
        st.text("JSON Data")
        st.json(response)