import math
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.shot_events import ShotJournal
from backend.ws import shots


def launch(journal, shot_id='one', **overrides):
    values=dict(shot_id=shot_id,speed=2,direction=95,forward=90,ppm=1000,
                calibration_source='manual',calibrated=True)
    return journal.append(**{**values,**overrides})


def test_journal_persists_identity_events_and_deduplicates(tmp_path):
    journal=ShotJournal(tmp_path/'events.db')
    first=launch(journal)
    assert launch(journal)==first
    reopened=ShotJournal(tmp_path/'events.db')
    assert reopened.device_id==journal.device_id
    assert reopened.session_id!=journal.session_id
    assert reopened.read(0)==[first]
    assert first['direction_deg']==5
    assert first['quality']['spin']=='unavailable'
    assert first['launch_angle_deg'] is None


@pytest.mark.parametrize('speed',[math.nan,math.inf,-1,0,101])
def test_rejects_invalid_launches(tmp_path,speed):
    journal=ShotJournal(tmp_path/'events.db')
    with pytest.raises(ValueError):launch(journal,speed=speed)
    assert journal.head()==0


def test_quality_does_not_invent_calibration(tmp_path):
    journal=ShotJournal(tmp_path/'events.db')
    assert launch(journal,calibrated=False)['quality']['speed']=='unavailable'
    assert launch(journal,'two',estimated=True)['quality']['speed']=='estimated'
    assert launch(journal,'three',ppm=1100)['calibration_id']!=launch(journal)['calibration_id']


def client_for(tmp_path,monkeypatch):
    journal=ShotJournal(tmp_path/'events.db')
    monkeypatch.setattr(shots,'get_shot_journal',lambda:journal)
    app=FastAPI();app.include_router(shots.router)
    return TestClient(app),journal


def test_fresh_subscriber_does_not_replay_old_shots(tmp_path,monkeypatch):
    client,journal=client_for(tmp_path,monkeypatch)
    launch(journal)
    with client.websocket_connect('/ws/shots') as ws:
        ws.send_json({'type':'subscribe','client_id':'browser-client'})
        assert ws.receive_json()['cursor']==1
        second=launch(journal,'two')
        assert ws.receive_json()==second
        ws.send_json({'type':'ack','seq':2})
        ws.send_json({'type':'ping'})
        assert ws.receive_json()=={'type':'pong'}


def test_reconnect_replays_unacknowledged_event_and_cursor_skips_applied(tmp_path,monkeypatch):
    client,journal=client_for(tmp_path,monkeypatch)
    event=launch(journal)
    for _ in range(2):
        with client.websocket_connect('/ws/shots') as ws:
            ws.send_json({'type':'subscribe','client_id':'browser-client','after':0})
            ws.receive_json()
            assert ws.receive_json()==event
    with client.websocket_connect('/ws/shots') as ws:
        ws.send_json({'type':'subscribe','client_id':'browser-client','after':1})
        assert ws.receive_json()['cursor']==1
        ws.send_json({'type':'ping'})
        assert ws.receive_json()['type']=='pong'


def test_invalid_cursor_is_closed(tmp_path,monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    client,_=client_for(tmp_path,monkeypatch)
    with client.websocket_connect('/ws/shots') as ws:
        ws.send_json({'type':'subscribe','client_id':'browser-client','after':-1})
        with pytest.raises(WebSocketDisconnect):ws.receive_json()
