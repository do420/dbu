from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.api_v1.endpoints import auth, processes
from db.session import engine
from db.base import Base  # This will import all models
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from api.api_v1.api import api_router
# Initialize database
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(title="AI Superapp Backend")


app.mount("/_OUTPUT", StaticFiles(directory="_OUTPUT"), name="_OUTPUT")
# Include routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(processes.router, prefix="/api/v1/processes")

app.add_middleware( 
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js çalıştığı origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
# Serve static files (CSS, JS, images)
# Serve the index page
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)