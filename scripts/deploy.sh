#!/bin/bash
set -e

cd /root/website

FORCE=false
if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
  FORCE=true
fi

BEFORE=$(git rev-parse HEAD)
git fetch origin
git reset --hard origin/master
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ] && [ "$FORCE" = false ]; then
  echo "No changes. Skipping build."
  exit 0
fi

if [ "$FORCE" = true ]; then
  echo "Force build requested."
fi

echo "Changes detected. Building..."
npm install
npm run build
rm -rf /var/www/html/*
cp -r dist/* /var/www/html
echo "Deploy complete."
