from models import get_model
    
def train(df):
    model = get_model()#model_name)
    model.train(df)
    # return trained_model