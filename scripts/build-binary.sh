#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Building site..."
npm run build

echo "Building binary..."
BUILD_DIR=$(mktemp -d)
cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

cp scripts/serve.go "$BUILD_DIR/"
cp -r dist "$BUILD_DIR/"
cd "$BUILD_DIR"

go mod init kgup.me/serve
go build -o website-serve serve.go

cd "$OLDPWD"
cp "$BUILD_DIR/website-serve" .
echo "Done: $(pwd)/website-serve"

echo ""
echo "Usage:"
echo "  ./website-serve          # serves on :8080"
echo "  PORT=3000 ./website-serve  # custom port"
echo "  ./website-serve -port 3000 # same"
