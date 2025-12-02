# **Lab 2.8: Building an Advanced Gemini Chat Application with Streamlit**

## **Learning Objectives**

By the end of this lab, students will be able to:

### **Core Competencies:**
1. **Integrate Gemini API** into a Python web application
2. **Design interactive UIs** using Streamlit for AI applications
3. **Implement multimodal AI** (text + image) in real applications
4. **Manage chat state** and session persistence
5. **Handle API errors** gracefully with user-friendly feedback

### **Technical Skills:**
6. **Configure multiple AI models** (text and vision) dynamically
7. **Implement streaming responses** and real-time updates
8. **Create data export functionality** (JSON, TXT)
9. **Apply responsive design principles** for AI applications
10. **Optimize API usage** with proper parameter tuning

### **Applied Knowledge:**
11. **Compare different Gemini models** for specific use cases
12. **Design prompt engineering interfaces**
13. **Build production-ready AI applications**
14. **Implement security best practices** for API keys
15. **Create reusable AI application patterns**

---

## **Step-by-Step Implementation Guide**

### **Step 1: Environment Setup (5 minutes)**

**Objective:** Prepare the development environment with all necessary dependencies.

```bash
# 1. Create project directory
mkdir lab-2-8-gemini-chat
cd lab-2-8-gemini-chat

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install required packages
# 4.1 Create 'requirements.txt':
streamlit>=1.28.0
google-generativeai>=0.3.0
pillow>=10.0.0
pandas>=2.0.0
plotly>=5.17.0
python-dotenv>=1.0.0

# 4.1 install the required Libraries
pip install -r requirements.txt

# 5. Create requirements.txt for documentation
pip freeze > requirements.txt

# 6. Create project structure
mkdir -p assets/images
touch app.py utils.py config.py
```

### **Step 2: API Configuration Setup (10 minutes)**

**Objective:** Set up secure API key management and configuration.

**File: `config.py`**
```python
"""
Simplified Configuration without dotenv
"""

import os

class Config:
    """Application configuration"""
    
    # API Configuration - Hardcode your API key here temporarily
    GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
    
    # Model configurations
    TEXT_MODELS = {
        "Gemini 2.5 Flash (Recommended)": "gemini-2.5-flash",
        "Gemini 2.0 Flash (Stable)": "gemini-2.0-flash",
        "Gemini 2.5 Pro (Advanced)": "gemini-2.5-pro",
        "Gemma 3 4B (Open Model)": "gemma-3-4b-it",
        "Gemini Flash Latest": "gemini-flash-latest",
    }
    
    MULTIMODAL_MODELS = {
        "Gemini 2.5 Flash Image": "gemini-2.5-flash-image",
        "Gemini 3 Pro Image Preview": "gemini-3-pro-image-preview",
    }
    
    # Default settings
    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1024
    
    # UI Settings
    PAGE_TITLE = "Gemini Chat Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    # File upload settings
    ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "gif", "webp"]
    MAX_FILE_SIZE_MB = 5
    
    @classmethod
    def validate_config(cls):
        """Validate configuration on startup"""
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            return False  # Indicate API key needs to be set
        return True
```

**File: `.env`**
```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_api_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO

# Optional: Custom settings
APP_TITLE="AI Chat Assistant"
```

### **Step 3: Utility Functions Creation (15 minutes)**

**Objective:** Create reusable helper functions for the application.

**File: `utils.py`**
```python
"""
Utility functions for Gemini Chat Application
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
from datetime import datetime
from typing import Optional, Dict, Any
import base64

class GeminiUtils:
    """Utility class for Gemini API operations"""
    
    @staticmethod
    def initialize_gemini(api_key: str):
        """Initialize Gemini API with configuration"""
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {str(e)}")
            return False
    
    @staticmethod
    def create_generation_config(temperature: float, max_tokens: int, 
                                top_p: float = 0.95, top_k: int = 40):
        """Create generation configuration dictionary"""
        return {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
    
    @staticmethod
    def generate_response(model_name: str, prompt: str, 
                         generation_config: Dict, 
                         image: Optional[Image.Image] = None):
        """Generate response from Gemini model"""
        try:
            model = genai.GenerativeModel(model_name)
            
            if image:
                content = [prompt, image]
            else:
                content = prompt
            
            response = model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(**generation_config)
            )
            
            return {
                "success": True,
                "text": response.text,
                "usage_metadata": getattr(response, 'usage_metadata', None)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": None
            }
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    @staticmethod
    def format_chat_history(messages: list) -> str:
        """Format chat history for display or export"""
        formatted = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            if timestamp:
                formatted.append(f"[{timestamp}] {role}: {content}")
            else:
                formatted.append(f"{role}: {content}")
        
        return "\n\n".join(formatted)
    
    @staticmethod
    def create_model_info_card(model_name: str) -> str:
        """Create informational card about selected model"""
        info = {
            "gemini-2.5-flash": {
                "description": "Fast, efficient model for general purpose tasks",
                "strengths": ["Speed", "Cost-effective", "General reasoning"],
                "best_for": ["Chat", "Summarization", "Code generation"],
                "limitations": ["Complex reasoning", "Creative writing"]
            },
            "gemini-2.5-pro": {
                "description": "Advanced model for complex reasoning tasks",
                "strengths": ["Reasoning", "Creativity", "Analysis"],
                "best_for": ["Analysis", "Creative writing", "Problem solving"],
                "limitations": ["Slower response", "Higher cost"]
            },
            "gemini-2.5-flash-image": {
                "description": "Multimodal model for image and text analysis",
                "strengths": ["Image understanding", "Visual reasoning"],
                "best_for": ["Image analysis", "Document parsing", "Visual Q&A"],
                "limitations": ["Text-only tasks"]
            },
            "gemma-3-4b-it": {
                "description": "Open, efficient model from Google",
                "strengths": ["Open source", "Efficient", "Transparent"],
                "best_for": ["Experimentation", "Education", "Lightweight apps"],
                "limitations": ["Smaller capacity", "Basic tasks only"]
            }
        }
        
        return info.get(model_name, {
            "description": "General purpose AI model",
            "strengths": ["Versatile", "Accessible"],
            "best_for": ["Various tasks"],
            "limitations": ["Check specific capabilities"]
        })
```

### **Step 4: Building the Main Application (30 minutes)**

**Objective:** Create the complete Streamlit application with all features.

**File: `app.py`**
```python
"""
Main Streamlit application for Gemini Chat Assistant
Lab 2.8: Building Advanced AI Applications
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd
from datetime import datetime
from typing import Optional
# Replace this import:
# from dotenv import load_dotenv

# With this try-except block:
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using hardcoded API key from config.py.")


# Import local modules
from config import Config
from utils import GeminiUtils

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
def load_custom_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 0 1rem;
        }
        
        /* Header styling */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        
        /* Model cards */
        .model-card {
            background: #1e1e2d;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
            margin: 0.5rem 0;
        }
        
        /* Chat messages */
        .user-message {
            background: #2b313e;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #4CAF50;
        }
        
        .assistant-message {
            background: #262730;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #2196F3;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* Metrics cards */
        .metric-card {
            background: #1e1e2d;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_start_time" not in st.session_state:
        st.session_state.chat_start_time = datetime.now()
    
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = Config.DEFAULT_MODEL

def create_sidebar():
    """Create the sidebar configuration panel"""
    with st.sidebar:
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.title("⚙️ Configuration")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Key Section
        with st.expander("🔑 API Configuration", expanded=True):
            api_key = st.text_input(
                "Gemini API Key:",
                value=Config.GEMINI_API_KEY or "",
                type="password",
                help="Get your API key from Google AI Studio"
            )
            
            if api_key:
                if GeminiUtils.initialize_gemini(api_key):
                    st.success("✅ API Configured")
                else:
                    st.error("❌ API Configuration Failed")
        
        # Model Selection Section
        with st.expander("🤖 Model Selection", expanded=True):
            model_type = st.radio(
                "Select Model Type:",
                ["Text Models", "Multimodal Models"],
                horizontal=True
            )
            
            if model_type == "Text Models":
                model_options = Config.TEXT_MODELS
            else:
                model_options = Config.MULTIMODAL_MODELS
            
            selected_model_display = st.selectbox(
                "Choose Model:",
                options=list(model_options.keys()),
                index=0
            )
            
            st.session_state.selected_model = model_options[selected_model_display]
            
            # Model information card
            model_info = GeminiUtils.create_model_info_card(
                st.session_state.selected_model
            )
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.caption("📊 Model Information")
            st.write(f"**Description:** {model_info['description']}")
            st.write(f"**Best for:** {', '.join(model_info['best_for'])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generation Parameters
        with st.expander("🎛️ Generation Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Temperature:",
                    min_value=0.0,
                    max_value=1.0,
                    value=Config.DEFAULT_TEMPERATURE,
                    step=0.1,
                    help="Controls randomness (0=deterministic, 1=creative)"
                )
            
            with col2:
                max_tokens = st.slider(
                    "Max Tokens:",
                    min_value=100,
                    max_value=4096,
                    value=Config.DEFAULT_MAX_TOKENS,
                    step=100,
                    help="Maximum length of response"
                )
            
            # Advanced parameters
            top_p = st.slider(
                "Top-P:",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.05,
                help="Nucleus sampling parameter"
            )
            
            top_k = st.slider(
                "Top-K:",
                min_value=1,
                max_value=100,
                value=40,
                help="Sample from top K tokens"
            )
        
        # Image Upload for Multimodal Models
        if model_type == "Multimodal Models":
            with st.expander("🖼️ Image Input", expanded=True):
                uploaded_file = st.file_uploader(
                    "Upload Image:",
                    type=Config.ALLOWED_IMAGE_TYPES,
                    help="Upload an image for analysis"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    st.session_state.uploaded_image = image
                else:
                    st.session_state.uploaded_image = None
        
        # Export and Actions
        with st.expander("💾 Export & Actions", expanded=True):
            if st.button("📥 Export Chat History", use_container_width=True):
                export_chat_history()
            
            if st.button("🔄 Reset Chat", use_container_width=True):
                reset_chat_session()
            
            if st.button("🧪 Test Connection", use_container_width=True):
                test_api_connection()
        
        # Statistics
        with st.expander("📈 Statistics", expanded=True):
            display_statistics()
        
        return {
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "model_type": model_type
        }

def display_chat_interface():
    """Display the main chat interface"""
    # Header
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.title("💬 Gemini Chat Assistant")
    st.caption(f"Using: {st.session_state.selected_model}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    with col2:
        if st.session_state.messages:
            avg_length = sum(len(m["content"]) for m in st.session_state.messages) / len(st.session_state.messages)
            st.metric("Avg Msg Length", f"{int(avg_length)} chars")
    with col3:
        duration = datetime.now() - st.session_state.chat_start_time
        st.metric("Session Duration", f"{duration.seconds // 60}m")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display message history
        for i, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f'<div class="user-message">**You:** {content}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">**Assistant:** {content}</div>', 
                           unsafe_allow_html=True)
            
            # Add model info for assistant messages
            if role == "assistant" and message.get("model"):
                st.caption(f"Model: {message['model']} | Tokens: {message.get('tokens', 'N/A')}")
    
    # Chat input
    st.divider()
    
    # Image preview for multimodal
    if st.session_state.uploaded_image:
        st.info(f"📎 Image ready for analysis with {st.session_state.selected_model}")
    # Use st.chat_input instead of text_area for better chat experience
    if prompt := st.chat_input("Type your message here..."):
        process_user_input(prompt)

    # Input area
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_area(
            "Your message:",
            placeholder="Type your message here...",
            height=100,
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("🚀", help="Send message", use_container_width=True):
            if user_input:
                process_user_input(user_input)

def process_user_input(user_input: str):
    """Process user input and generate response"""
    # Don't modify widget state, just process the input
    if not user_input.strip():
        return  # Don't process empty input
    
    # Add user message to history
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    }
    
    if st.session_state.uploaded_image:
        user_message["has_image"] = True
    
    st.session_state.messages.append(user_message)
    
    # Generate response
    with st.spinner("Generating response..."):
        generation_config = GeminiUtils.create_generation_config(
            st.session_state.get("temperature", Config.DEFAULT_TEMPERATURE),
            st.session_state.get("max_tokens", Config.DEFAULT_MAX_TOKENS),
            st.session_state.get("top_p", 0.95),
            st.session_state.get("top_k", 40)
        )
        
        # Prepare content (with image if available)
        image = st.session_state.uploaded_image if st.session_state.get("model_type") == "Multimodal Models" else None
        
        result = GeminiUtils.generate_response(
            model_name=st.session_state.selected_model,
            prompt=user_input,
            generation_config=generation_config,
            image=image
        )
        
        if result["success"]:
            # Add assistant response to history
            assistant_message = {
                "role": "assistant",
                "content": result["text"],
                "model": st.session_state.selected_model,
                "timestamp": datetime.now().isoformat()
            }
            
            # Track token usage
            if result.get("usage_metadata"):
                tokens = result["usage_metadata"].total_token_count
                assistant_message["tokens"] = tokens
                st.session_state.total_tokens += tokens
            
            st.session_state.messages.append(assistant_message)
            
            # Instead of modifying widget state, use rerun to refresh
            st.rerun()
        else:
            st.error(f"Error: {result['error']}")
            provide_troubleshooting_help(result['error'])

def provide_troubleshooting_help(error: str):
    """Provide context-sensitive troubleshooting help"""
    with st.expander("🔧 Troubleshooting Help"):
        if "API key" in error or "authentication" in error:
            st.error("**API Key Issue:**")
            st.markdown("""
            1. Check if API key is entered correctly
            2. Ensure key has not expired
            3. Verify key has necessary permissions
            """)
        elif "quota" in error.lower():
            st.warning("**Quota Exceeded:**")
            st.markdown("""
            1. Check your quota at [Google AI Studio Dashboard](https://makersuite.google.com/app/dashboard)
            2. Consider using a smaller model (Gemini Flash instead of Pro)
            3. Reduce request frequency
            """)
        elif "model" in error.lower() and "found" in error.lower():
            st.info("**Model Not Available:**")
            st.markdown("""
            1. Try switching to 'gemini-flash-latest'
            2. Check model availability in your region
            3. Use the model test button to verify
            """)
        else:
            st.info("**General Tips:**")
            st.markdown("""
            1. Try simplifying your prompt
            2. Reduce temperature for more consistent results
            3. Check your internet connection
            4. Try again in a few minutes
            """)

def export_chat_history():
    """Export chat history in multiple formats"""
    if not st.session_state.messages:
        st.warning("No chat history to export")
        return
    
    # Create JSON export
    chat_data = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "model": st.session_state.selected_model,
            "total_messages": len(st.session_state.messages),
            "total_tokens": st.session_state.total_tokens
        },
        "messages": st.session_state.messages
    }
    
    json_str = json.dumps(chat_data, indent=2, default=str)
    
    # Create text export
    text_str = GeminiUtils.format_chat_history(st.session_state.messages)
    
    # Provide download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        st.download_button(
            label="📄 Download Text",
            data=text_str,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def reset_chat_session():
    """Reset the chat session"""
    st.session_state.messages = []
    st.session_state.chat_start_time = datetime.now()
    st.session_state.total_tokens = 0
    st.session_state.uploaded_image = None
    st.success("Chat session reset successfully!")
    st.rerun()

def test_api_connection():
    """Test API connection with current configuration"""
    if not Config.GEMINI_API_KEY:
        st.error("Please configure API key first")
        return
    
    with st.spinner("Testing connection..."):
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model = genai.GenerativeModel(st.session_state.selected_model)
            response = model.generate_content("Hello", generation_config={
                "max_output_tokens": 10
            })
            
            st.success(f"✅ Connection successful! Model: {st.session_state.selected_model}")
            st.info(f"Test response: {response.text}")
            
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

def display_statistics():
    """Display chat statistics"""
    if st.session_state.messages:
        # Basic stats
        user_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        assistant_count = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        st.metric("User Messages", user_count)
        st.metric("Assistant Messages", assistant_count)
        st.metric("Total Tokens", st.session_state.total_tokens)
        
        # Token efficiency
        if assistant_count > 0:
            avg_tokens = st.session_state.total_tokens // assistant_count
            st.metric("Avg Tokens/Response", avg_tokens)
    else:
        st.info("No messages yet. Start a conversation!")

def main():
    """Main application function"""
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar and get configuration
    config = create_sidebar()
    
    # Store configuration in session state
    for key, value in config.items():
        if key != "api_key":  # Don't store API key in session
            st.session_state[key] = value
    
    # Display chat interface
    display_chat_interface()
    
    # Footer
    st.divider()
    st.caption(f"Lab 2.8 • Gemini Chat Assistant • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    # Validate configuration
    try:
        Config.validate_config()
        main()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please create a .env file with GEMINI_API_KEY=your_key")
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Check the console for detailed error information")
```

### **Step 5: Testing and Deployment (10 minutes)**

**Objective:** Test the application and prepare for deployment.

```bash
# 1. Run the application locally
streamlit run app.py

# 2. Test different features:
#    - Text-only chat
#    - Image upload with multimodal models
#    - Model switching
#    - Export functionality

# 3. Create deployment configuration
touch streamlit_config.toml

# File: streamlit_config.toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

# 4. Deploy to Streamlit Cloud (Optional)
#    - Push to GitHub repository
#    - Connect at share.streamlit.io
#    - Add GEMINI_API_KEY as secrets
```

### **Step 6: Advanced Features Extension (Optional)**

Add these features to enhance the application:

1. **Streaming Responses:**
```python
# Add to process_user_input function
response_stream = model.generate_content(
    content,
    stream=True
)

response_text = ""
message_placeholder = st.empty()
for chunk in response_stream:
    response_text += chunk.text
    message_placeholder.markdown(response_text)
```

2. **Prompt Templates:**
```python
PROMPT_TEMPLATES = {
    "Summarize": "Please summarize the following text:\n\n{text}",
    "Translate": "Translate to {language}:\n\n{text}",
    "Code Review": "Review this code and suggest improvements:\n\n{code}",
}
```

3. **Conversation Analysis:**
```python
def analyze_conversation():
    """Analyze conversation patterns"""
    topics = model.generate_content(
        "Extract main topics from this conversation: " + 
        GeminiUtils.format_chat_history(st.session_state.messages)
    )
    return topics.text
```
## **Troubleshooting Guide**

| Issue | Solution |
|-------|----------|
| API Key Error | Check .env file, verify key in Google AI Studio |
| Model Not Found | Use `gemini-flash-latest` or check available models |
| Image Upload Fails | Verify file type, size, and use multimodal model |
| Slow Responses | Reduce max_tokens, use Flash model instead of Pro |
| Memory Issues | Implement message limit, clear session periodically |
| Export Not Working | Check JSON serialization, file permissions |

## **Further Exploration**

1. **Add voice input/output** using speech recognition/synthesis
2. **Implement conversation memory** with vector databases
3. **Create specialized agents** for different tasks
4. **Add user authentication** for multi-user support
5. **Implement rate limiting** and usage tracking
6. **Create plugin system** for extending functionality
7. **Add analytics dashboard** for usage insights
8. **Implement A/B testing** for different models/prompts
