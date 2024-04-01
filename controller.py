#!/usr/bin/env python

import asyncio
import websockets
import predict

async def server():
    print("Server Started")
    async with websockets.serve(predict.classify, "", 8001):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(server())