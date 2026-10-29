#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# mdBook 0.5.4 does not accept --config-file, so we swap configs temporarily.

echo "Building English book..."
mdbook build

echo "Building Chinese book..."
cp book.toml book.toml.en
cp book.zh.toml book.toml
mdbook build -d book/zh
mv book.toml.en book.toml

echo "Done. Outputs: book/ and book/zh/"
