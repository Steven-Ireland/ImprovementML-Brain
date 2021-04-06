import asyncio
import websockets

BASE_PORT = 7776
NUM_BINDS = 8

def get_handler():
    clients = set()
    async def broadcast(source, msg):
        await asyncio.gather(
            *[ws.send(msg) for ws in clients if ws != source],
            return_exceptions=True,
        )
    async def handler(websocket, path):
        clients.add(websocket)
        try:
            async for msg in websocket:
                try:
                    await broadcast(websocket, msg)
                except:
                    print("Heyo, an error ignored baybe")
        finally:
            clients.remove(websocket)
    return handler

try:

    for port in range(BASE_PORT, BASE_PORT + NUM_BINDS):
        start_server = websockets.serve(get_handler(), "localhost", port)
        asyncio.get_event_loop().run_until_complete(start_server)

    asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
    asyncio.get_event_loop().stop()