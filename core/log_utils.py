from models.log import Log
from sqlalchemy.orm import Session
from datetime import datetime

def create_log(db: Session, user_id: int, log_type: int, description: str, ip_address: str = None):
    log = Log(
        ip_address=ip_address,
        type=log_type, #0 : create mini service, 1: create agent, 2: enhance system prompt of agent,  3: remove agent, 4: delete mini service, 5: run mini service, 6: file upload, 7 : delete chat conversation, 8: create chat conversation
        description=description,
        created_at=datetime.utcnow(),
        user_id=user_id  # Add this field to Log model if not present
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log