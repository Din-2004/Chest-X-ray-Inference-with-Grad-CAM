import http.server
import socketserver
import os
PORT = 4173
ROOT = r"C:\Users\Dinesh mungale\Desktop\X_RAY"
os.chdir(ROOT)
Handler = http.server.SimpleHTTPRequestHandler
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
with ReusableTCPServer(("127.0.0.1", PORT), Handler) as httpd:
    httpd.serve_forever()
