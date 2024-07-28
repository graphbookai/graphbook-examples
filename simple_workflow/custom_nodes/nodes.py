from graphbook.steps import Step
from graphbook import Note
import random

class MyFirstStep(Step):
    RequiresInput = True
    Parameters = {
        "prob": {
            "type": "resource"
        }
    }
    Outputs = ["A", "B"]
    Category = "Custom"
    def __init__(self, id, logger, prob):
        super().__init__(id, logger)
        self.prob = prob

    def on_after_items(self, note: Note) -> Note:
        self.logger.log(note['message'])

    def forward_note(self, note: Note) -> str:
        if random.random() < self.prob:
            return "A"
        return "B"
    
from graphbook.steps import SourceStep

class MyFirstSource(SourceStep):
    RequiresInput = False
    Parameters = {
        "message": {
            "type": "string",
            "default": "Hello, World!"
        }
    }
    Outputs = ["message"]
    Category = "Custom"
    def __init__(self, id, logger, message):
        super().__init__(id, logger)
        self.message = message

    def load(self):
        return {
            "message": [Note({"message": self.message}) for _ in range(10)]
        }

    def forward_note(self, note: Note) -> str:
        return "message"
