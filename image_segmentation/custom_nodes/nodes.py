from graphbook.steps import BatchStep, SourceStep
from graphbook.resources import Resource
from graphbook import Note
from transformers import AutoModelForImageSegmentation
import torchvision.transforms.functional as F
import torch.nn.functional
import torch
from typing import List
from PIL import Image
import os
import os.path as osp

class RMBGModel(Resource):
    Category = "Custom"
    Parameters = {
        "model_name": {
            "type": "string",
            "description": "The name of the image processor.",
        }
    }

    def __init__(self, model_name: str):
        super().__init__(
            AutoModelForImageSegmentation.from_pretrained(
                model_name, trust_remote_code=True
            ).to("cuda")
        )


class RemoveBackground(BatchStep):
    RequiresInput = True
    Parameters = {
        "model": {
            "type": "resource",
        },
        "batch_size": {
            "type": "number",
            "default": 8,
        },
        "item_key": {
            "type": "string",
            "default": "image",
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
        model: AutoModelForImageSegmentation,
    ):
        super().__init__(id, logger, batch_size, item_key)
        self.model = model

    @staticmethod
    def load_fn(item: dict) -> torch.Tensor:
        im = Image.open(item["value"])
        image = F.to_tensor(im)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]

        return image

    @staticmethod
    def dump_fn(t: torch.Tensor, output_dir: str, uid: int):
        img = F.to_pil_image(t)
        img.save(osp.join(output_dir, f"{uid}.png"))

    @torch.no_grad()
    def on_item_batch(
        self, tensors: List[torch.Tensor], items: List[dict], notes: List[Note]
    ):
        og_sizes = [t.shape[1:] for t in tensors]

        images = [
            F.normalize(
                torch.nn.functional.interpolate(
                    torch.unsqueeze(image, 0), size=[1024, 1024], mode="bilinear"
                ),
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            )
            for image in tensors
        ]
        images = torch.stack(images).to("cuda")
        images = torch.squeeze(images, 1)
        tup = self.model(images)
        result = tup[0][0]
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        resized = [
            torch.squeeze(
                torch.nn.functional.interpolate(
                    torch.unsqueeze(image, 0), size=og_size, mode="bilinear"
                ),
                0,
            )
            for image, og_size in zip(result, og_sizes)
        ]
        return {"removed_bg": resized}

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
