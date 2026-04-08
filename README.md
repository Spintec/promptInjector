Here's what you've got now:


promptInjector/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── tester.py
├── run.sh
└── custom_payloads_example.json
Three ways to run it
1. Docker against a remote API (Claude, Grok, Gemini):


export API_KEY="sk-ant-..."
export URL="https://api.anthropic.com/v1/messages"
export PRESET="claude"
./run.sh docker
2. Docker against Ollama on your host machine:


export URL="http://host.docker.internal:11434/api/generate"
export PRESET="ollama"
./run.sh docker
host.docker.internal bridges from the container to your host's port 11434.

3. Docker with a bundled Ollama (zero host dependencies):


./run.sh docker-with-ollama
This starts an Ollama container, pulls llama3 on first run, then runs the tests against it. Fully self-contained — works on any OS with Docker.

Results land in ./results/results.json for Docker modes, or ./results.json for bare-metal. Extra args pass through — e.g. ./run.sh docker --tests 3 --output /app/results/out.csv.