from abc import ABC, abstractmethod
import os

class BaseModel(ABC):
    def __init__(self):
        # self.embedding_model = os.getenv("EMBEDDING_MODEL")
        pass

    @abstractmethod
    def train(self, train_file):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, test_file, named_entities):
        """Predict roles and sub-roles."""
        pass
