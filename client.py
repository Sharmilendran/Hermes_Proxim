#!/usr/bin/env python

import asyncio
import websockets

async def client():
    async with websockets.connect("ws://localhost:8001") as websocket:
            await websocket.send("start")
            while True:
                message = await websocket.recv()
                print(message)


if __name__ == "__main__":
    asyncio.run(client())