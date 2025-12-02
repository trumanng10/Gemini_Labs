# **Labs 2.7 to 2.10: Building Apps with Gemini API**

## **Lab 2.7: Making Your First API Call**
**Learning Objectives:**
- Understand the difference between Google AI Studio (UI) and Gemini API (programmatic)
- Set up API credentials and Python environment
- Make a basic API call and handle the response
- Learn rate limits and basic error handling

**Step-by-Step: Building a Simple CLI App**

### **Step 1: Get Your API Key**
1. In Google AI Studio homepage, click **"Get API Key"** in sidebar
2. Create a new API key (copy and save securely)

### **Step 2: Set Up Python Environment**
```bash
# Create project folder
mkdir gemini-app
cd gemini-app

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required packages
pip install google-generativeai python-dotenv
```

### **Step 3: Create Your First App**
**File: `simple_chat.py`**
```python
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file (create .env with GOOGLE_API_KEY=your_key_here)
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

def chat_with_gemini():
    print("Gemini CLI Assistant (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        try:
            # Generate response
            response = model.generate_content(user_input)
            print(f"\nGemini: {response.text}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_gemini()
```

### **Step 4: Run and Test**
```bash
python simple_chat.py
```

---

## **Lab 2.8: Building a Streamlit Web Chatbot**
**Learning Objectives:**
- Create interactive web apps with Streamlit
- Implement chat memory and conversation history
- Add UI controls for model parameters (temperature, max tokens)
- Deploy a functional web interface

### **Step 1: Install Additional Dependencies**
```bash
pip install streamlit
```

### **Step 2: Create Streamlit App**
**File: `chatbot_app.py`**
```python
import streamlit as st
import google.generativeai as genai
import os

# Page configuration
st.set_page_config(
    page_title="Gemini Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
    
    model_name = st.selectbox(
        "Choose Model:",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    )
    
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens:", 100, 2000, 500)

# Main chat interface
st.title("💬 Gemini Chat Assistant")
st.caption("Powered by Google's Gemini AI")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    if not api_key:
        st.error("Please enter your API key in the sidebar!")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                response = model.generate_content(prompt)
                st.markdown(response.text)
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.text
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Add clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
```

### **Step 3: Run the App**
```bash
streamlit run chatbot_app.py
```

---

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

---

## **Lab 2.10: Multimodal Image Analysis Dashboard**
**Learning Objectives:**
- Build an app that processes both images and text
- Implement file upload and preview functionality
- Create analysis pipelines for different image types
- Generate structured reports from visual content

### **Step 1: Install Additional Dependencies**
```bash
pip install pillow python-multipart plotly
```

### **Step 2: Create Multimodal App**
**File: `image_analyzer.py`**
```python
import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
import plotly.graph_objects as go
import os

# App configuration
st.set_page_config(
    page_title="Multimodal Image Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Functions
def analyze_image_with_gemini(image, prompt, api_key):
    """Send image and prompt to Gemini for analysis"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-vision')
    
    # Prepare the content
    content = [prompt, image]
    
    # Generate response
    response = model.generate_content(content)
    return response.text

def extract_json_from_response(response_text):
    """Try to extract JSON from Gemini response"""
    try:
        # Look for JSON-like content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    return None

# Sidebar
with st.sidebar:
    st.title("🔧 Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("Analysis Type")
    analysis_type = st.selectbox(
        "Choose Analysis",
        ["General Description", "Technical Analysis", "Creative Writing", 
         "Object Detection", "Text Extraction", "Custom Prompt"]
    )
    
    # Custom prompt input
    if analysis_type == "Custom Prompt":
        custom_prompt = st.text_area("Enter your analysis prompt:", 
                                     "Describe this image in detail and identify key elements.")
    else:
        # Predefined prompts
        prompts = {
            "General Description": "Provide a detailed description of this image. Identify main subjects, colors, composition, and overall mood.",
            "Technical Analysis": "Analyze this image technically. Discuss lighting, perspective, possible camera settings, and composition techniques.",
            "Creative Writing": "Write a creative story or poem inspired by this image. Include emotional elements and narrative.",
            "Object Detection": "List all objects visible in this image. For each object, provide confidence level and location description.",
            "Text Extraction": "Extract all readable text from this image. Preserve formatting and structure if possible."
        }
        custom_prompt = prompts[analysis_type]
    
    st.subheader("Advanced Options")
    include_json = st.checkbox("Request structured JSON output", value=True)
    if include_json:
        json_format = st.text_area("JSON Schema (optional)", 
                                   '{"description": "", "objects": [], "colors": [], "analysis": ""}')

# Main interface
st.title("🔍 Multimodal Image Analyzer")
st.caption("Upload an image and let Gemini analyze it")

# File upload
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png', 'gif', 'bmp']
)

if uploaded_file and api_key:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.caption(f"Size: {image.size[0]}x{image.size[1]} | Format: {image.format}")
    
    with col2:
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Analyzing with Gemini..."):
                try:
                    # Prepare prompt
                    final_prompt = custom_prompt
                    if include_json and json_format:
                        final_prompt += f"\n\nPlease format your response as valid JSON with this structure: {json_format}"
                    
                    # Analyze
                    analysis_result = analyze_image_with_gemini(image, final_prompt, api_key)
                    
                    # Display results
                    st.subheader("📋 Analysis Results")
                    
                    # Try to parse as JSON
                    json_data = extract_json_from_response(analysis_result)
                    
                    if json_data:
                        # Display as JSON and structured data
                        with st.expander("Raw JSON Data", expanded=True):
                            st.json(json_data)
                        
                        # Create visualizations if we have structured data
                        if "objects" in json_data and len(json_data["objects"]) > 0:
                            st.subheader("📊 Detected Objects")
                            objects_df = pd.DataFrame(json_data["objects"])
                            st.dataframe(objects_df)
                            
                            # Simple chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(range(len(json_data["objects"]))),
                                    y=[1]*len(json_data["objects"]),  # Placeholder for confidence
                                    text=[obj.get("name", "") for obj in json_data["objects"]]
                                )
                            ])
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Display as text
                        st.write(analysis_result)
                    
                    # Save to history
                    st.session_state.analysis_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "image_name": uploaded_file.name,
                        "analysis_type": analysis_type,
                        "result": analysis_result[:500] + "..." if len(analysis_result) > 500 else analysis_result
                    })
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

# History section
if st.session_state.analysis_history:
    with st.expander("📚 Analysis History", expanded=False):
        for idx, item in enumerate(reversed(st.session_state.analysis_history)):
            st.write(f"**{idx+1}. {item['image_name']}** ({item['analysis_type']})")
            st.caption(f"Time: {item['timestamp']}")
            st.code(item['result'][:200] + "...", language='text')
            st.divider()

# Export functionality
if st.button("💾 Export All Results"):
    if st.session_state.analysis_history:
        with open("image_analysis_history.json", "w") as f:
            json.dump(st.session_state.analysis_history, f, indent=2)
        st.success("Results exported to image_analysis_history.json")
```

### **Step 3: Run the Dashboard**
```bash
streamlit run image_analyzer.py
```

---

## **Deployment Guide (All Apps)**

### **Option 1: Local Deployment**
```bash
# Run any app locally
streamlit run app_name.py --server.port 8501 --server.address 0.0.0.0
```

### **Option 2: Deploy to Streamlit Cloud (Free)**
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add API key as secrets:
   ```python
   # In your app, replace hardcoded API key with:
   import streamlit as st
   api_key = st.secrets["GOOGLE_API_KEY"]
   ```

### **Option 3: Deploy as Python Package**
Create `requirements.txt`:
```
google-generativeai
streamlit
python-dotenv
pillow
pandas
plotly
```

### **Security Best Practices:**
1. **Never hardcode API keys** in your source code
2. Use environment variables or secret management
3. Implement rate limiting for production apps
4. Add input validation and sanitization
5. Consider adding user authentication for sensitive apps

---

**Next Steps for Students:**
1. Start with Lab 2.7 to understand API basics
2. Progress through each lab, building on previous knowledge
3. Customize each app with their own features
4. Deploy at least one app publicly
5. Explore error handling and edge cases

Would you like me to provide additional labs focusing on specific use cases (e.g., educational tools, business applications, creative projects)?
