import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    service_name = "service"

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        payload = json.dumps(
            {"status": "ok", "service": self.service_name}
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        return


def start_health_server(port: int, service_name: str):
    handler = type(
        "ServiceHealthHandler",
        (HealthHandler,),
        {"service_name": service_name},
    )
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
