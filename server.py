import sys
from pathlib import Path

# Add v-ui directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "v-ui"))

from server import run_server

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
