#!/usr/bin/env python3
"""
Qwen 3.8-27B Local Streaming Server (Pure C Backend)
Provides SSE streaming HTTP endpoints for the web interface.
"""

import http.server
import json
import os
import subprocess
import threading
import urllib.parse

PORT = 8080
BIN_PATH = os.path.join(os.path.dirname(__file__), "build", "qwen")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Qwen3.8-27B-IQ4_XS.gguf")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

current_process = None
process_lock = threading.Lock()


class QwenServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        global current_process
        url = urllib.parse.urlparse(self.path)

        if url.path == "/api/stop":
            with process_lock:
                if current_process and current_process.poll() is None:
                    current_process.terminate()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "stopped"}')
            return

        if url.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            prompt = data.get("prompt", "")
            temp = str(data.get("temp", "0.0"))
            top_p = str(data.get("top_p", "0.9"))
            top_k = str(data.get("top_k", "40"))
            repeat_penalty = str(data.get("repeat_penalty", "1.1"))
            n_predict = str(data.get("n_predict", "256"))
            threads = str(data.get("threads", "12"))
            model_file = data.get("model", MODEL_PATH)

            cmd = [
                BIN_PATH,
                "-m", model_file,
                "-p", prompt,
                "-n", n_predict,
                "-t", threads,
                "--temp", temp,
                "--top-p", top_p,
                "--top-k", top_k,
                "--repeat-penalty", repeat_penalty,
                "--spec", "1"
            ]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with process_lock:
                if current_process and current_process.poll() is None:
                    current_process.terminate()
                try:
                    current_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=1,
                        universal_newlines=False
                    )
                except Exception as e:
                    err_payload = f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    self.wfile.write(err_payload.encode("utf-8"))
                    return

            proc = current_process
            header_passed = False

            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break

                try:
                    decoded = line.decode("utf-8", errors="replace")
                except Exception:
                    continue

                if not header_passed:
                    if "Generation:" in decoded:
                        header_passed = True
                        after = decoded.split("Generation:", 1)[1]
                        if after:
                            payload = f"data: {json.dumps({'type': 'token', 'content': after})}\n\n"
                            self.wfile.write(payload.encode("utf-8"))
                            self.wfile.flush()
                    continue

                if line.startswith(b"===") or "[spec]" in decoded or "Prefill" in decoded:
                    continue

                if decoded:
                    payload = f"data: {json.dumps({'type': 'token', 'content': decoded})}\n\n"
                    try:
                        self.wfile.write(payload.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        proc.terminate()
                        break

            proc.wait()
            done_payload = f"data: {json.dumps({'type': 'done'})}\n\n"
            try:
                self.wfile.write(done_payload.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen 3.8-27B Web Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind (default: 8080)")
    args = parser.parse_args()

    server_address = ("", args.port)
    httpd = http.server.ThreadingHTTPServer(server_address, QwenServerHandler)

    print("=" * 60)
    print("  Qwen 3.8-27B Inference Server (C + AVX2 + GDN + MTP)")
    print("=" * 60)
    print(f"  Web Interface : http://localhost:{args.port}")
    print(f"  Model         : {MODEL_PATH}")
    print(f"  Engine Binary : {BIN_PATH}")
    print("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
