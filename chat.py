import panel as pn
from ctransformers import AutoModelForCausalLM

pn.extension()

llms = {}

SYSTEM_INSTRUCTIONS = "You are 'RigCheck AI', a professional hardware appraiser for used gaming PCs. Provide realistic 'Quick Sell' vs 'Market Value' prices. Rules: 1. Be realistic for 2025. 2. Ask for missing GPU/CPU details. 3. Format in a Markdown table. 4. Suggest an upgrade path."

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if "mistral" not in llms:
        instance.placeholder_text = "Powering up RigCheck AI..."
        llms["mistral"] = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            gpu_layers=1, 
        )
    
    llm = llms["mistral"]
    
    full_prompt = f"<s>[INST] {SYSTEM_INSTRUCTIONS}\n\nUser Request: {contents} [/INST]"
    
    response = llm(full_prompt, stream=True, max_new_tokens=500, temperature=0.2)
    
    message = ""
    for token in response:
        message += token
        yield message

chat_interface = pn.chat.ChatInterface(
    callback=callback, 
    callback_user="RigCheck AI",
    show_clear=True,  
    show_undo=True,  
    width=700
)

chat_interface.send(
    "Welcome to RigCheck AI! Paste your PC specs (CPU, GPU, RAM) and I'll give you a 2025 market appraisal.", 
    user="System", 
    respond=False
)

chat_interface.servable()

if __name__ == "__main__":
    pn.serve(chat_interface)