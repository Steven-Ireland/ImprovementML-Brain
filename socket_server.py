import asyncio
import websockets

CLIENTS = set()

async def broadcast(source, msg):
    await asyncio.gather(
        *[ws.send(msg) for ws in CLIENTS if ws != source],
        return_exceptions=False,
    )

async def handler(websocket, path):
    CLIENTS.add(websocket)
    try:
        async for msg in websocket:
            await broadcast(websocket, msg)
    finally:
        CLIENTS.remove(websocket)


try:
    start_server = websockets.serve(handler, "localhost", 7777)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
    asyncio.get_event_loop().stop()