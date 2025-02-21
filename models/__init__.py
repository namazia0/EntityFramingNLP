#from .bert import BERT
from .roberta import RoBERTa
from .bert import BERT

def get_model(model_name):
    models = {
        "roberta": RoBERTa,
        "bert": BERT
    }
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]
