"""Write fp16 and int8 variants of the embedder next to the fp32 model.

A variant may only ship if parity_test.py passes with --onnx <variant>.

    python scripts/hasta/quantize.py
"""

import os

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from hasta_ref import DEFAULT_MODEL_DIR

SRC = os.path.join(DEFAULT_MODEL_DIR, "hasta_dinov2_vits14_embed.onnx")


def main():
    fp16_path = SRC.replace(".onnx", "_fp16.onnx")
    # keep_io_types: the app still feeds float32 "crop" and reads float32 "embedding"
    onnx.save(convert_float_to_float16(onnx.load(SRC), keep_io_types=True), fp16_path)

    int8_path = SRC.replace(".onnx", "_int8.onnx")
    quantize_dynamic(SRC, int8_path, weight_type=QuantType.QInt8)

    for p in (SRC, fp16_path, int8_path):
        print(f"{os.path.basename(p):45s} {os.path.getsize(p) / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
