"""Launch the standalone built game. No cameras or simulator setup required."""
import argparse
import functools
import http.server
from pathlib import Path
import socket
import threading
import webbrowser
import json
import urllib.request


class GameHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        route=self.path.split('?')[0]
        if route=='/__grenland_health':
            body=b'{"app":"strikelab-grenland","status":"ready"}'
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            return
        if route=='/':
            self.send_response(302);self.send_header('Location','/play/grenland');self.end_headers()
            return
        if route in ('/play/grenland', '/play/grenland/'):
            self.path='/index.html'
        super().do_GET()

    def list_directory(self,path):
        self.send_error(404)
        return None

    def end_headers(self):
        if self.path.split('?')[0] in ('/sw.js','/index.html'):
            self.send_header('Cache-Control','no-cache')
        super().end_headers()

    def guess_type(self,path):
        if path.endswith('.f32') or path.endswith('.u8'):return 'application/octet-stream'
        if path.endswith('.webmanifest'):return 'application/manifest+json'
        return super().guess_type(path)


class GameServer(http.server.ThreadingHTTPServer):
    # Windows SO_REUSEADDR can allow two listeners on the same port.
    allow_reuse_address=False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8088)
    parser.add_argument('--lan',action='store_true',help='Allow phones on this local network')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    dist=Path(__file__).resolve().parent/'frontend/dist'
    if not (dist/'index.html').is_file():raise SystemExit('Build the game first: npm --prefix frontend run build')
    url=f'http://127.0.0.1:{args.port}/play/grenland'
    handler=functools.partial(GameHandler,directory=str(dist))
    try:server=GameServer(('0.0.0.0' if args.lan else '127.0.0.1',args.port),handler)
    except OSError as error:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/__grenland_health',timeout=2) as response:
                existing=json.load(response)
            if existing.get('app')=='strikelab-grenland':
                print(f'Grenland is already running: {url}',flush=True)
                if not args.no_browser:webbrowser.open(url)
                return
        except (OSError,ValueError):pass
        raise SystemExit(f'Port {args.port} is unavailable. Use --port with another port. {error}')
    print(f'Play Grenland: {url}',flush=True)
    if args.lan:
        print(f'Phone address: http://{socket.gethostbyname(socket.gethostname())}:{args.port}/play/grenland',flush=True)
    if not args.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
