import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Embedding Llama chatbot with PDFs.')
parser.add_argument('--pdf_file', type=str, default='ipb4.pdf', help='The PDF file to load')
parser.add_argument('--temperature', type=float, default=0.1, help='The temperature for the LlamaCPP model')
parser.add_argument('--model_path', type=str, default='/Users/lukasz/Desktop/StoryNook/data/llamaModels/smaller.gguf', help='The path to the LlamaCPP model')
args = parser.parse_args()
