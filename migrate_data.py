import pandas as pd
from database import SessionLocal, Appointment, init_db
from datetime import datetime
import sqlalchemy

def migrate():
    # 1. Initialize the tables
    init_db()
    db = SessionLocal()

    # 1. Read the CSV first  (So 'df' is always defined, even if the file is missing or empty)
    df = pd.read_csv("doctor_availability.csv")

    # PATCH: Check if data already exists to prevent duplicates
    existing_count = db.query(Appointment).count()
    if existing_count > 0:
        print(f"⚠️ Database already contains {existing_count} records. Skipping migration.")
        db.close()
        return

    
    # 2. Read the existing CSV
#    df = pd.read_csv("doctor_availability.csv")

    for _, row in df.iterrows():
        # Convert string to proper Python datetime object
        dt_obj = pd.to_datetime(row['date_slot'])

        # PATCH: Use .strip() and .lower() to ensure data is clean for the AI to search
        new_appointment = Appointment(
            date_slot=dt_obj,
            specialization=row['specialization'].strip().lower(),
            doctor_name=row['doctor_name'].strip().lower(),
            is_available=bool(row['is_available']),
            # Ensure patient_id is handled as a string or None
            patient_id=str(int(row['patient_to_attend'])) if pd.notna(row['patient_to_attend']) else None
        )
        db.add(new_appointment)
        
    db.commit()
    db.close()
    print(" Data migrated from CSV to Postgres successfully!")

if __name__ == "__main__":
    migrate()