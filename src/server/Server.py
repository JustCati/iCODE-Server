import ssl
import http.server

from src.queue.FrameQueue import FrameQueue
from src.queue.BatchQueue import BatchQueue


class Server():
    def __init__(self, key_file, cert_file,  ip = "0.0.0.0", port=8443, visor_callback_port=4444, max_queue_size=10000):
        self.ip = ip
        self.port = port
        self.KEY_FILE = key_file
        self.CERT_FILE = cert_file
        self.max_queue_size = max_queue_size

        self.last_client_ip = None
        self.visor_callback_port = visor_callback_port

        self.handler_class = self.FrameRequestHandler
        self.httpd = http.server.HTTPServer((self.ip, self.port), self.handler_class)

        self.frame_queue = FrameQueue(max_queue_size)
        self.batch_queue = BatchQueue(self.frame_queue, max_queue_size)

        self.handler_class.batch_queue = self.batch_queue
        self.httpd.server_instance = self


    class FrameRequestHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            print()
            print("Received POST request.")
            content_length = int(self.headers.get('Content-Length', 0))
            frame_data = self.rfile.read(content_length)
            client_ip = self.client_address[0]
            print(f"Received frame ({len(frame_data)} bytes).")

            self.send_response(200)
            self.end_headers()
            self.flush_headers()
            self.wfile.write(b"OK")
            print("Sent response.")

            self.server.server_instance.last_client_ip = client_ip
            self.__class__.batch_queue.put(frame_data)


    def run(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        context.load_cert_chain(certfile=self.CERT_FILE, keyfile=self.KEY_FILE)
        self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)

        print(f"Starting HTTPS server on ip {self.ip} and port {self.port}...", end="\n\n")
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            self.httpd.server_close()
            self.batch_queue.batch_worker_thread.join()
            print("Server has been shut down.")


    def get_frame_queue(self):
        return self.frame_queue
