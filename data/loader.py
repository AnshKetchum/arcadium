from arcadium.utils import load_config
from arcadium.data.single_folder import DocumentLanguageModelDatasetFromShardsRandomSampling

def load_language_dataset(config_path: str):
    # Load data configuration
    conf = load_config(config_path)
    
    
    