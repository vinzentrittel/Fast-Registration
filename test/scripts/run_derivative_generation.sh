#!/usr/bin/env bash
# generate_derivatives.sh
# Usage: ./generate_derivatives.sh <input_atlas.stl> <output_directory>
set -e

INPUT_STL="$1"
OUTPUT_DIR="$2"

if [[ -z "$INPUT_STL" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <input_atlas.stl> <output_directory>"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

blender --background --python generate_derivatives.py -- \
  --input "${INPUT_STL}" \
  --output "${OUTPUT_DIR}" \

echo "Erzeugung von  Derivaten abgeschlossen."

