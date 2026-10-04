"""Local-only integration fixture. Synthetic events; never connects to cameras."""
import asyncio
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, Field
import uvicorn

from backend.shot_events import ShotJournal
from backend.ws import shots

app=FastAPI()
journal=ShotJournal(Path(tempfile.mkdtemp(prefix='grenland-shot-test-'))/'events.db')
shots.get_shot_journal=lambda:journal
app.include_router(shots.router)


class Launch(BaseModel):
    shot_id:str=Field(default_factory=lambda:str(uuid.uuid4()))
    speed:float=Field(default=1.5,gt=0,le=12)
    direction:float=0


@app.post('/test/launch')
def launch(event:Launch):
    return journal.append(shot_id=event.shot_id,speed=event.speed,direction=event.direction,
                          forward=0,ppm=1000,calibration_source='SYNTHETIC TEST FIXTURE',
                          calibrated=True,estimated=True)


@app.get('/test/state')
def state():
    with journal.connect() as db:
        consumers=db.execute('SELECT client_id,ack_seq FROM consumers').fetchall()
    return {'synthetic':True,'head':journal.head(),'consumers':consumers}


@app.get('/api/health')
def health():
    return {'status':'ready','server_up':True,'camera_ready':False,'startup_phase':'synthetic-test'}


@app.websocket('/ws')
async def state_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({'v':2,'t':'snapshot','seq':1,'ts_ms':0,'payload':{
                'state':'ARMED','forward_direction_deg':0,'ready_status':'test',
                'ball_visible':False,'pixels_per_meter':1000}})
            await asyncio.sleep(.1)
    except Exception:pass


if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=8000)
