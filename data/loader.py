from utils import load_config
from models.tasks.language.datasets.single_file import DocumentLanguageModelDatasetFromFileRandomSampling

def load_language_dataset(config_path: str):
    # Load data configuration
    conf = load_config(config_path)
    
    
    