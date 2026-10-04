"""The standalone server must not take over an occupied Windows port."""
import functools
import json
import threading
import urllib.error
import urllib.request

import pytest
from play_grenland import GameHandler, GameServer


@pytest.fixture
def game_server(tmp_path):
    (tmp_path/'index.html').write_text('Grenland test build')
    (tmp_path/'sw.js').write_text('// test worker')
    handler=functools.partial(GameHandler,directory=str(tmp_path))
    server=GameServer(('127.0.0.1',0),handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield server,handler
    server.shutdown();server.server_close();thread.join(timeout=2)


def test_game_launch_routes_and_cache_headers(game_server):
    server,_=game_server;root=f'http://127.0.0.1:{server.server_port}'
    with urllib.request.urlopen(root+'/') as response:
        assert response.url.endswith('/play/grenland')
        assert response.read()==b'Grenland test build'
        assert response.headers['Cache-Control']=='no-cache'
    with urllib.request.urlopen(root+'/__grenland_health') as response:
        assert json.load(response)['app']=='strikelab-grenland'
    with urllib.request.urlopen(root+'/sw.js') as response:
        assert response.headers['Cache-Control']=='no-cache'


def test_second_server_cannot_bind_the_same_port(game_server):
    server,handler=game_server
    with pytest.raises(OSError):GameServer(('127.0.0.1',server.server_port),handler)
