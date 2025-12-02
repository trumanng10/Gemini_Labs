## **Lab 2.9: Content Generator with Multiple Output Formats**
**Learning Objectives:**
- Build a multi-functional content generator
- Implement file export functionality (TXT, PDF, JSON)
- Use structured outputs for different content types
- Add batch processing capabilities

### **Step 1: Create Advanced Content App**
**File: `content_generator.py`**
```python
import streamlit as st
import google.generativeai as genai
import json
import os
from datetime import datetime
import pandas as pd

# App setup
st.set_page_config(page_title="AI Content Generator", layout="wide")

# Initialize Gemini
@st.cache_resource
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

# Content templates
TEMPLATES = {
    "blog_post": "Write a comprehensive blog post about: {topic}. Include an engaging title, introduction, 3 main sections with subheadings, and a conclusion. Target audience: {audience}",
    "social_media": "Create 5 social media posts for {platform} about: {topic}. Include relevant hashtags and emojis. Tone: {tone}",
    "email": "Write a professional email about: {topic}. Purpose: {purpose}. Include subject line, greeting, body, and closing. Length: {length}",
    "product_description": "Create a compelling product description for: {product}. Highlight 3 key features, benefits, and include a call-to-action.",
    "meeting_agenda": "Generate a detailed meeting agenda for: {meeting_topic}. Include date/time, participants, objectives, agenda items with time allocations, and action items."
}

def generate_content(template, **kwargs):
    """Generate content based on template and parameters"""
    prompt = TEMPLATES[template].format(**kwargs)
    model = st.session_state.model
    response = model.generate_content(prompt)
    return response.text

# Main app
st.title("📝 AI Content Generator")
st.sidebar.header("Configuration")

# API Key input
api_key = st.sidebar.text_input("Gemini API Key", type="password")
if api_key:
    st.session_state.model = init_gemini(api_key)

# Content type selection
content_type = st.selectbox("Select Content Type", list(TEMPLATES.keys()))

# Dynamic form based on content type
with st.form("content_form"):
    if content_type == "blog_post":
        topic = st.text_input("Blog Topic", "The Future of AI in Education")
        audience = st.selectbox("Target Audience", ["Students", "Educators", "Tech Professionals", "General Public"])
    
    elif content_type == "social_media":
        topic = st.text_input("Campaign Topic", "New Product Launch")
        platform = st.selectbox("Platform", ["Twitter", "Instagram", "LinkedIn", "Facebook"])
        tone = st.selectbox("Tone", ["Professional", "Casual", "Funny", "Inspirational"])
    
    # Add other content type forms...
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
    with col2:
        variants = st.slider("Number of Variants", 1, 5, 1)
    
    submitted = st.form_submit_button("Generate Content")

# Generate and display content
if submitted and api_key:
    with st.spinner(f"Generating {content_type}..."):
        try:
            # Generate multiple variants if requested
            all_content = []
            for i in range(variants):
                content = generate_content(
                    content_type,
                    topic=topic,
                    audience=audience if 'audience' in locals() else None,
                    platform=platform if 'platform' in locals() else None,
                    tone=tone if 'tone' in locals() else None
                )
                all_content.append(content)
            
            # Display results
            st.subheader("Generated Content")
            for idx, content in enumerate(all_content):
                with st.expander(f"Variant {idx + 1}"):
                    st.write(content)
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📥 Copy Text {idx+1}"):
                            st.write("Copied to clipboard!")
                    with col2:
                        if st.button(f"📝 Save as TXT {idx+1}"):
                            with open(f"content_variant_{idx+1}.txt", "w") as f:
                                f.write(content)
                            st.success("File saved!")
                    with col3:
                        if st.button(f"📊 Analyze {idx+1}"):
                            # Simple analysis
                            word_count = len(content.split())
                            st.info(f"Word Count: {word_count}")
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

```
* streamlit run content-generator.py
