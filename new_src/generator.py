from ollama import generate
class Generator:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def generate(self, prompt):
        return generate(model=self.model_name, prompt=prompt)