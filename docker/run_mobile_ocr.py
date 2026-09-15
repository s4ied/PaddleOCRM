#!/usr/bin/env python3
"""Detection + recognition only, using the lightweight mobile models."""

import argparse
import json
import platform
from pathlib import Path

from paddleocr import PaddleOCR

# Recognition model per language. Detection is language independent.
REC_MODELS = {
    "en": "en_PP-OCRv5_mobile_rec",
    "de": "latin_PP-OCRv5_mobile_rec",
    "latin": "latin_PP-OCRv5_mobile_rec",
    "ch": "PP-OCRv5_mobile_rec",
}

DET_MODEL = "PP-OCRv5_mobile_det"


def default_enable_mkldnn() -> bool:
    # oneDNN segfaults in Paddle's aarch64 CPU build.
    return platform.machine().lower() not in ("aarch64", "arm64")


def build_ocr(
    lang: str,
    device: str | None = None,
    enable_mkldnn: bool | None = None,
    cpu_threads: int = 2,
) -> PaddleOCR:
    if enable_mkldnn is None:
        enable_mkldnn = default_enable_mkldnn()
    return PaddleOCR(
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODELS[lang],
        # No rotation handling: skip doc orientation, unwarping and textline orientation.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_rec_score_thresh=0.5,
        device=device,
        enable_mkldnn=enable_mkldnn,
        cpu_threads=cpu_threads,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Image file, PDF or directory of images.")
    parser.add_argument("-l", "--lang", default="de", choices=sorted(REC_MODELS))
    parser.add_argument("-o", "--output", default="./output")
    parser.add_argument(
        "--device",
        default=None,
        help="e.g. cpu, gpu:0. Defaults to the PaddleOCR auto-selection.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also write visualisations of the detected boxes.",
    )
    parser.add_argument(
        "--mkldnn",
        dest="enable_mkldnn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force oneDNN on/off. Defaults to off on ARM, on elsewhere.",
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_ocr(args.lang, args.device, args.enable_mkldnn, args.cpu_threads)

    for result in ocr.predict_iter(args.input):
        texts = result["rec_texts"]
        scores = result["rec_scores"]
        boxes = result["rec_polys"]

        print(json.dumps({"input": result["input_path"], "lines": len(texts)}))
        for text, score, box in zip(texts, scores, boxes):
            x0, y0 = box[0]
            print(f"  [{score:.3f}] ({int(x0)},{int(y0)}) {text}")

        result.save_to_json(str(out_dir))
        if args.save_images:
            result.save_to_img(str(out_dir))


if __name__ == "__main__":
    main()
