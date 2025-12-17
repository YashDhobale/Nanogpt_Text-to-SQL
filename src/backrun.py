# Run this file to start the FastAPI Backend server
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NanoGPT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print("\nRegistered FastAPI routes:")
    for route in app.routes:
        print("  ROUTE:", route.path, getattr(route, 'methods', 'N/A'))

    # Start server
    print("\nStarting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)