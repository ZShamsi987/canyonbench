#!/usr/bin/env python
"""Serve the non-language detector reference on Lambda.

This is the Section 12 upper reference: a segmenter that reports what the
imagery supports independent of language, so a VLM failure at 16 km can be
attributed to the imagery or to the model. It answers the same structured
schema every model answers, over the `/predict` contract the http_detector
adapter speaks.

It deliberately uses only the standard library plus the serving stack that vLLM
already installs (torch, transformers, pillow) — no extra dependency, and it is
small enough (<8 GB) to sit alongside the VLM session.

    python scripts/lambda/serve_detector.py --port 8010

Everything decidable without a GPU lives in `canyonbench.detector` and is unit
tested; this file is the thin serving shell around it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

from canyonbench.detector import (
    DetectorQuery,
    parse_detector_prompt,
    resolve_label_ids,
    response_from_segmentation,
)

LOGGER = logging.getLogger("canyonbench.detector")
DEFAULT_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"


class Segmenter:
    """Semantic segmentation backend, loaded once and reused per request."""

    def __init__(self, checkpoint: str, device: str) -> None:
        import torch
        from transformers import (  # type: ignore[import-not-found]
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.torch = torch
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
        self.model.eval().to(device)
        self.device = device
        self.id2label = {int(key): str(value) for key, value in self.model.config.id2label.items()}
        # Fail at startup, not mid-run, if a registered class has no label.
        self.label_ids = {
            feature: resolve_label_ids(self.id2label, feature)
            for feature in ("water", "road", "field")
        }
        for feature, ids in self.label_ids.items():
            LOGGER.info("%s -> %s", feature, [self.id2label[i] for i in ids])

    def segment(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        upsampled = self.torch.nn.functional.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def _handler(segmenter: Segmenter) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            LOGGER.debug(fmt, *args)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in {"", "/health"}:
                self._send(200, {"status": "ok", "labels": segmenter.label_ids})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/predict":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                query: DetectorQuery = parse_detector_prompt(str(payload.get("prompt", "")))
                encoded = str(payload["image_base64"])
                # The adapter may send a data: URL or bare base64.
                if "," in encoded[:64]:
                    encoded = encoded.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            except Exception as error:
                self._send(400, {"error": f"bad request: {error}"})
                return
            try:
                segmentation = segmenter.segment(image)
                response = response_from_segmentation(
                    segmentation,
                    segmenter.label_ids[query.target_class],
                    query,
                )
            except Exception as error:
                LOGGER.exception("inference failed")
                self._send(500, {"error": str(error)})
                return
            self._send(200, {"response": response})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args()
    logging.basicConfig(level=arguments.log_level, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("loading %s on %s", arguments.checkpoint, arguments.device)
    segmenter = Segmenter(arguments.checkpoint, arguments.device)
    server = ThreadingHTTPServer((arguments.host, arguments.port), _handler(segmenter))
    LOGGER.info("detector reference listening on http://%s:%d", arguments.host, arguments.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
