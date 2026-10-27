"""Start a real local HTTP server, wait for model warmup, run cagent examples, stop."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx


def run_demo(output: Path, startup_timeout: float, model: Optional[str] = None) -> int:
    """Keep logs and actual responses; always terminate only our child server."""
    root = Path(__file__).resolve().parent
    env = dict(os.environ)
    env.setdefault("DECISION_API_KEY", secrets.token_urlsafe(32))
    if model:
        env["MODEL"] = model
    result = {"status": "starting", "backend": env.get("BACKEND", "rbln-compile")}
    process = None
    try:
        # Reserve an actual socket and pass it to uvicorn: no port collision race.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            result["url"] = url
            with (output / "server.log").open("w") as server_log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "server:create_app",
                        "--factory",
                        "--fd",
                        str(listener.fileno()),
                        "--workers",
                        "1",
                    ],
                    cwd=root,
                    env=env,
                    pass_fds=(listener.fileno(),),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )
                print(f"Starting model server; logs: {output / 'server.log'}", flush=True)
                start = time.monotonic()
                next_progress = start + 15
                with httpx.Client(timeout=2, trust_env=False) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"Server exited ({process.returncode}); inspect server.log"
                            )
                        if time.monotonic() - start > startup_timeout:
                            raise TimeoutError("Model readiness timeout; inspect server.log")
                        try:
                            response = client.get(url + "/readyz")
                            if response.status_code == 200:
                                break
                            if "startup_failed" in response.text:
                                raise RuntimeError(
                                    "Model loading/compilation failed; inspect server.log"
                                )
                        except httpx.RequestError:
                            pass
                        if time.monotonic() >= next_progress:
                            print(
                                f"Waiting for readiness: {time.monotonic() - start:.0f}s",
                                flush=True,
                            )
                            next_progress = time.monotonic() + 15
                        time.sleep(0.25)
                result["startup_seconds"] = time.monotonic() - start
                print(f"Server ready: {url}; calling sync and async cagent examples...", flush=True)
                with (output / "client.log").open("w") as client_log:
                    run = subprocess.run(
                        [
                            sys.executable,
                            str(root / "cagent_example.py"),
                            "--base-url",
                            url,
                            "--output",
                            str(output / "responses.json"),
                        ],
                        cwd=root,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                if run.returncode:
                    raise RuntimeError("cagent example failed; inspect client.log")
                result["status"] = "completed"
                print((output / "responses.json").read_text(), flush=True)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        (output / "demo-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"HTTP demo results: {output}")
    return 0 if result["status"] == "completed" else 1


def main() -> int:
    """Use a fresh output folder and a generous first-compile readiness budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error("startup-timeout must be positive")
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        base = Path(__file__).resolve().parent / "results"
        base.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="http-demo-", dir=base))
    return run_demo(output, args.startup_timeout, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
