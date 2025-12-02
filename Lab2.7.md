# **Labs 2.7: Building Apps with Gemini API**

*Check Available Model in Gemini*
https://github.com/trumanng10/Gemini_Labs/blob/main/list_model.py

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
model = genai.GenerativeModel('gemini-2.5-flash')

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

