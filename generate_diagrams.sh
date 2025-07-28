#!/bin/bash

# Create directories if they don't exist
mkdir -p docs/diagrams
mkdir -p docs/images

# Generate PNGs for each diagram
for file in docs/diagrams/*.mmd; do
    filename=$(basename "$file" .mmd)
    echo "Generating PNG for $filename..."
    mmdc -i "$file" -o "docs/images/${filename}.png" -b transparent
done

echo "All diagrams have been generated!" 