#!/usr/bin/env bash
# generate_derivatives.sh
# Usage: ./generate_derivatives.sh <input_atlas.stl> <output_directory>
set -e

INPUT_STL="$1"
OUTPUT_DIR="$2"
NUM_DERIV=1000

if [[ -z "$INPUT_STL" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <input_atlas.stl> <output_directory>"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

for i in $(seq 1 ${NUM_DERIV}); do
  # Formatierung der Ausgabe-Datei
  OUT="${OUTPUT_DIR}/derivative_$(printf "%04d" "$i").stl"

  blender --background --python generate_derivatives.py -- \
    --input "${INPUT_STL}" \
    --output "${OUT}" \
    --seed "${i}"
done

echo "Erzeugung von ${NUM_DERIV} Derivaten abgeschlossen."

