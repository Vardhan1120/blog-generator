import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # Load the model (Ensure the path is correct and use raw string for Windows paths)
        llm = CTransformers(
            model=r'C:\blog generator\models\llama-2-7b-chat.ggmlv3.q8_0.bin',  # Use raw string for Windows path
            model_type='llama',
            config={'max_new_tokens': 256, 'temperature': 0.01}
        )
        
        # Define the prompt template
        template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
        """
        
        # Create prompt from template
        prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'], template=template)
        
        # Generate the response using the model
        response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        return response
    
    except Exception as e:
        print(f"Error while generating response: {e}")
        return "Error: Could not generate response. Please check the model file or configuration."

# Streamlit configuration
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

# Streamlit UI
st.header("Generate Blogs 🤖")

# User inputs for blog generation
input_text = st.text_input("Enter the Blog Topic")

# Create two columns for additional inputs
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

# Button to generate the blog
submit = st.button("Generate")

# Display the final response and download option when the button is pressed
if submit:
    blog_content = getLLamaresponse(input_text, no_words, blog_style)
    st.write(blog_content)
    
    # Adding a download button for the generated blog content
    if blog_content and not blog_content.startswith("Error:"):
        # Set the filename and file content
        file_name = f"{input_text.replace(' ', '_')}_blog.txt"
        
        # Provide download button to download the content as a .txt file
        st.download_button(
            label="Download Blog as Text File",
            data=blog_content,
            file_name=file_name,
            mime="text/plain"
        )
