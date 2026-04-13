from database import SessionLocal, Appointment
from datetime import datetime
from sqlalchemy import func

def check_slot_availability_db(doctor_name: str, date_slot: str):
    """
    Checks the database to see if a specific doctor is free at a specific time.
    Args:
        doctor_name: The name of the dentist (e.g., 'Emily Johnson')
        date_slot: The time string in 'M/D/YYYY H:MM' format
    """
    db = SessionLocal()
    try:
        # Convert string input to a python datetime object
        # This handles the "09:00 vs 9:00" issue automatically!
        target_date = datetime.strptime(date_slot, "%m/%d/%Y %H:%M")
        print("DB DEBUG:")
        print("Input doctor:", repr(doctor_name))
        print("Normalized:", doctor_name.strip().lower())
        print("Target date:", target_date)

        slot = db.query(Appointment).filter(
            func.trim(func.lower(Appointment.doctor_name))== doctor_name.strip().lower(),
            Appointment.date_slot == target_date
        ).first()
        all_slots = db.query(Appointment).all()
        for s in all_slots:
            print(f"DB SLOT: Doctor={s.doctor_name}, Date={s.date_slot}, Available={s.is_available}, Patient={s.patient_id}")

        if slot and slot.is_available and slot.patient_id is None:
            print("✅ SLOT AVAILABLE")
            return {
                "found": True,
                "message": f"Slot is available for {doctor_name} at {date_slot}."
            }
        print("❌ SLOT NOT AVAILABLE")
        return {"found": False, "message": f"No slot found for {doctor_name} at {date_slot}."}
    finally:
        db.close()