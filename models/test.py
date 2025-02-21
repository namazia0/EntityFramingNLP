from models import get_model

def test(model_name, df, test_file): # named_entities
    """
    Assigns a role and sub-role based on the model predictions.
    """
    model = get_model(model_name)
    return model.predict(df, test_file)