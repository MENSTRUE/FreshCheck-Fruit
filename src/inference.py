from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = ROOT / "models" / "freshcheck_effnet_b0_best.pt"
DEFAULT_CONFIG_PATH = ROOT / "reports" / "inference_config.json"
DEFAULT_RESULT_DIR = ROOT / "tests" / "results"


DEFAULT_CLASS_NAMES = [
    "Damaged",
    "Dried",
    "Old",
    "Ripe",
    "Unripe",
]

DEFAULT_IMAGE_SIZE = 224

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    top3: list[dict[str, float]]
    model_name: str
    image_size: int
    source_name: str | None = None
    saved_image: str | None = None
    saved_json: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_existing(
    data: dict[str, Any],
    keys: list[str],
    default: Any,
) -> Any:
    for key in keys:
        if key in data:
            return data[key]

    return default


class FreshCheckClassifier:
    """
    FreshCheck visual-quality classifier.

    Supported visual-quality classes:
    - Damaged
    - Dried
    - Old
    - Ripe
    - Unripe

    Important:
    This model performs visual classification only.

    It does NOT determine:
    - food safety,
    - chemical freshness,
    - shelf life,
    - whether food is safe to consume.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | None = None,
    ) -> None:

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self.model_path}"
            )

        self.config = _load_json(self.config_path)

        self.class_names = _first_existing(
            self.config,
            [
                "class_names",
                "classes",
                "labels",
            ],
            DEFAULT_CLASS_NAMES,
        )

        self.image_size = int(
            _first_existing(
                self.config,
                [
                    "image_size",
                    "img_size",
                    "input_size",
                ],
                DEFAULT_IMAGE_SIZE,
            )
        )

        self.mean = _first_existing(
            self.config,
            [
                "mean",
                "normalize_mean",
                "normalization_mean",
            ],
            DEFAULT_MEAN,
        )

        self.std = _first_existing(
            self.config,
            [
                "std",
                "normalize_std",
                "normalization_std",
            ],
            DEFAULT_STD,
        )

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = self._build_model()
        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.image_size,
                        self.image_size,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

    def _build_model(self) -> torch.nn.Module:
        model = models.efficientnet_b0(
            weights=None
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = torch.nn.Linear(
            in_features,
            len(self.class_names),
        )

        return model

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> dict[str, torch.Tensor]:

        if isinstance(checkpoint, dict):

            candidates = [
                "model_state_dict",
                "state_dict",
                "model",
                "weights",
            ]

            for key in candidates:
                value = checkpoint.get(key)

                if isinstance(value, dict):
                    return value

            if all(
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            ):
                return checkpoint

        raise RuntimeError(
            "Format checkpoint tidak dikenali."
        )

    @staticmethod
    def _clean_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        cleaned = {}

        prefixes = [
            "module.",
            "_orig_mod.",
            "model.",
        ]

        for key, value in state_dict.items():
            new_key = key

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            cleaned[new_key] = value

        return cleaned

    def _load_checkpoint(self) -> None:

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False,
        )

        # If metadata exists in checkpoint, prefer it.
        if isinstance(checkpoint, dict):

            checkpoint_classes = (
                checkpoint.get("class_names")
                or checkpoint.get("classes")
            )

            if checkpoint_classes is not None:
                if list(checkpoint_classes) != list(
                    self.class_names
                ):
                    print(
                        "[WARN] checkpoint class_names berbeda "
                        "dengan config. Menggunakan config order."
                    )

        state_dict = self._extract_state_dict(
            checkpoint
        )

        state_dict = self._clean_state_dict(
            state_dict
        )

        missing, unexpected = (
            self.model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        if unexpected:
            print(
                "[WARN] Unexpected checkpoint keys:",
                unexpected,
            )

        important_missing = [
            key
            for key in missing
            if "num_batches_tracked"
            not in key
        ]

        if important_missing:
            print(
                "[WARN] Missing checkpoint keys:",
                important_missing,
            )

    def predict(
        self,
        image: Image.Image | str | Path,
        source_name: str | None = None,
    ) -> PredictionResult:

        if not isinstance(
            image,
            Image.Image,
        ):
            image = Image.open(image)

        image = image.convert("RGB")

        tensor = self.transform(
            image
        ).unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        with torch.inference_mode():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        probabilities_cpu = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

        predicted_index = int(
            np.argmax(
                probabilities_cpu
            )
        )

        confidence = float(
            probabilities_cpu[
                predicted_index
            ]
        )

        probability_map = {
            class_name: float(
                probabilities_cpu[i]
            )
            for i, class_name
            in enumerate(
                self.class_names
            )
        }

        sorted_indices = np.argsort(
            probabilities_cpu
        )[::-1]

        top3 = [
            {
                "class": self.class_names[
                    int(index)
                ],
                "confidence": float(
                    probabilities_cpu[
                        int(index)
                    ]
                ),
            }
            for index
            in sorted_indices[:3]
        ]

        return PredictionResult(
            predicted_class=self.class_names[
                predicted_index
            ],
            predicted_index=predicted_index,
            confidence=confidence,
            probabilities=probability_map,
            top3=top3,
            model_name="EfficientNet-B0",
            image_size=self.image_size,
            source_name=source_name,
        )

    def save_result(
        self,
        original_image: Image.Image,
        result: PredictionResult,
        output_dir: str | Path = DEFAULT_RESULT_DIR,
    ) -> PredictionResult:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        safe_class = (
            result.predicted_class
            .lower()
            .replace(" ", "_")
        )

        base_name = (
            f"{timestamp}_{safe_class}"
        )

        prediction_image_path = (
            output_dir
            / f"{base_name}_prediction.jpg"
        )

        json_path = (
            output_dir
            / f"{base_name}_result.json"
        )

        image = (
            original_image
            .convert("RGB")
            .copy()
        )

        draw = ImageDraw.Draw(
            image
        )

        label = (
            f"{result.predicted_class} "
            f"({result.confidence:.1%})"
        )

        # Simple readable overlay.
        padding = 12

        try:
            bbox = draw.textbbox(
                (0, 0),
                label,
            )

            text_width = (
                bbox[2] - bbox[0]
            )

            text_height = (
                bbox[3] - bbox[1]
            )

        except Exception:
            text_width = 300
            text_height = 30

        draw.rectangle(
            [
                10,
                10,
                10
                + text_width
                + padding * 2,
                10
                + text_height
                + padding * 2,
            ],
            fill="black",
        )

        draw.text(
            (
                10 + padding,
                10 + padding,
            ),
            label,
            fill="white",
        )

        image.save(
            prediction_image_path,
            quality=95,
        )

        result.saved_image = str(
            prediction_image_path
        )

        result.saved_json = str(
            json_path
        )

        json_path.write_text(
            json.dumps(
                asdict(result),
                indent=2,
            ),
            encoding="utf-8",
        )

        self._append_history(
            result,
            output_dir,
        )

        return result

    @staticmethod
    def _append_history(
        result: PredictionResult,
        output_dir: Path,
    ) -> None:

        history_path = (
            output_dir
            / "prediction_history.csv"
        )

        exists = history_path.exists()

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            if not exists:

                writer.writerow(
                    [
                        "timestamp",
                        "source_name",
                        "predicted_class",
                        "confidence",
                        "model",
                        "saved_image",
                        "saved_json",
                    ]
                )

            writer.writerow(
                [
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    result.source_name,
                    result.predicted_class,
                    result.confidence,
                    result.model_name,
                    result.saved_image,
                    result.saved_json,
                ]
            )


if __name__ == "__main__":

    print(
        "FreshCheck inference module."
    )

    print(
        "Model:",
        DEFAULT_MODEL_PATH,
    )

    print(
        "Config:",
        DEFAULT_CONFIG_PATH,
    )

    print(
        "Results:",
        DEFAULT_RESULT_DIR,
    )