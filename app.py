from fastapi import FastAPI
import asyncio
app=FastAPI()
@app.get("/greetings")
async def greetings():
    await asyncio.sleep(10)
    return {"message":"Hello World"}