"""Local launch stream with explicit subscription and reconnect cursors."""
import asyncio
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..shot_events import get_shot_journal

router = APIRouter()


@router.websocket('/ws/shots')
async def shots(websocket: WebSocket):
    await websocket.accept()
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), 10)
        client = hello.get('client_id', '')
        if hello.get('type') != 'subscribe' or not isinstance(client, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}', client):
            await websocket.close(code=1008, reason='A stable client_id is required')
            return
        journal = get_shot_journal()
        head = journal.head()
        cursor = hello.get('after', head)
        if type(cursor) is not int or not 0 <= cursor <= head:
            await websocket.close(code=1008, reason='Invalid resume cursor')
            return
        await websocket.send_json({'type': 'subscribed', 'cursor': cursor, 'head': head,
                                   'device_id': journal.device_id})
        sent = cursor
        while True:
            # One outstanding batch; resend on reconnect until the consumer commits.
            if sent == cursor:
                for event in journal.read(cursor):
                    await websocket.send_json(event)
                    sent = event['seq']
            try:
                command = await asyncio.wait_for(websocket.receive_json(), 0.25)
            except asyncio.TimeoutError:
                continue
            if command.get('type') == 'ack':
                seq = command.get('seq')
                if type(seq) is not int or not cursor <= seq <= sent:
                    await websocket.close(code=1008, reason='Invalid acknowledgement')
                    return
                journal.acknowledge(client, seq)
                cursor = seq
            elif command.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except (ValueError, TypeError, AttributeError):
        await websocket.close(code=1008, reason='Invalid launch stream message')
