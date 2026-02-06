#!/bin/bash

cd "$1"

jupyter-book toc from-project .
jupyter-book build .

cd _build/html
python3 -m http.server 8000 &
SERVER_PID=$!
cd ../..

trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

echo "Server started at http://localhost:8000"
echo "Watching for changes... (Press Ctrl+C to stop)"

watchexec -e ipynb,md,yml,py -w . -i _build -i .git -i __pycache__ -i .ipynb_checkpoints -- jupyter-book build .
