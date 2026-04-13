# database.py (Connection and ORM setup for PostgreSQL)
# We’ll use SQLAlchemy to talk to the DB. This replaces your pandas CSV logic.
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://admin:password123@localhost:5432/dental_clinic"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True) # e.g., "doctor_date_time"
    date_slot = Column(DateTime, nullable=False)
    doctor_name = Column(String, nullable=False)
    specialization = Column(String, nullable=False)
    is_available = Column(Boolean, default=True)
    patient_id = Column(String, nullable=True)

# Create the tables
def init_db():
    Base.metadata.create_all(bind=engine)
