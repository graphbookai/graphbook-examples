from graphbook.steps import BatchStep, SourceStep
from graphbook.resources import Resource
from graphbook import Note
import os
import os.path as osp
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torchvision.transforms.functional as F
from PIL import Image
from typing import List


class PokemonClassifier(BatchStep):
    RequiresInput = True
    Parameters = {
        "batch_size": {"type": "number", "default": 8},
        "item_key": {"type": "string", "default": "image"},
        "model": {
            "type": "resource",
        },
        "image_processor": {
            "type": "resource",
        },
    }
    Outputs = ["out"]
    Category = "Custom"

    def __init__(
        self,
        id,
        logger,
        batch_size,
        item_key,
        model: ViTForImageClassification,
        image_processor: ViTImageProcessor,
    ):
        super().__init__(id, logger, batch_size, item_key)
        self.model = model
        self.image_processor = image_processor
        self.tp = 0
        self.num_samples = 0

    @staticmethod
    def load_fn(item: dict) -> torch.Tensor:
        im = Image.open(item["value"])
        image = F.to_tensor(im)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        return image

    @torch.no_grad()
    def on_item_batch(
        self, tensors: List[torch.Tensor], items: List[dict], notes: List[Note]
    ):
        extracted = self.image_processor(
            images=tensors, do_rescale=False, return_tensors="pt"
        )
        extracted = extracted.to("cuda")
        predicted_id = self.model(**extracted).logits.argmax(-1)
        for t, item, note in zip(predicted_id, items, notes):
            item["prediction"] = self.model.config.id2label[t.item()]
            self.logger.log(f"Predicted {item['value']} as {item['prediction']}")
            if item["prediction"] == note["name"]:
                self.tp += 1
            self.num_samples += 1
        if self.num_samples > 0:
            self.logger.log(f"Accuracy: {self.tp/self.num_samples:.2f}")


class LoadImageDataset(SourceStep):
    RequiresInput = False
    Outputs = ["out"]
    Category = "Custom"
    Parameters = {"image_dir": {"type": "string", "default": "/data/pokemon"}}

    def __init__(self, id, logger, image_dir: str):
        super().__init__(id, logger)
        self.image_dir = image_dir

    def load(self):
        subdirs = os.listdir(self.image_dir)

        def create_note(subdir):
            image_dir = osp.join(self.image_dir, subdir)
            return Note(
                {
                    "name": subdir,
                    "image": [
                        {"value": osp.join(image_dir, img), "type": "image"}
                        for img in os.listdir(image_dir)
                    ],
                }
            )

        return {"out": [create_note(subdir) for subdir in subdirs]}


class ViTForImageClassificationResource(Resource):
    Category = "Huggingface/Transformers"
    Parameters = {
        "model_name": {
            "type": "string",
            "description": "The name of the model to load.",
        }
    }

    def __init__(self, model_name: str):
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model = self.model.to("cuda")
        super().__init__(self.model)


class ViTImageProcessorResource(Resource):
    Category = "Huggingface/Transformers"
    Parameters = {
        "image_processor": {
            "type": "string",
            "description": "The name of the image processor.",
        }
    }

    def __init__(self, image_processor: str):
        self.image_processor = ViTImageProcessor.from_pretrained(image_processor)
        super().__init__(self.image_processor)
