from database import SessionLocal, Appointment
from datetime import datetime

def update_appointment_status_db(doctor_name: str, date_slot: str, patient_id: str = None, is_available: bool = False):
    """
    Updates the appointment status in the database.
    - To BOOK: Set is_available=False and provide a patient_id.
    - To CANCEL: Set is_available=True and patient_id=None.
    """
    db = SessionLocal()
    try:
        # Normalize the date string to a Python datetime object
        target_date = datetime.strptime(date_slot, "%m/%d/%Y %H:%M")

        print("🔍 DEBUG:")
        print("Doctor input:", doctor_name)
        print("Date input:", target_date)
        
        # Find the specific slot
        slot = db.query(Appointment).filter(
            Appointment.doctor_name == doctor_name,
            Appointment.date_slot == target_date
        ).first()

        if slot:
            print("✅ SLOT FOUND")
            slot.is_available = is_available
            slot.patient_id = patient_id
            db.commit()
            print("✅ SLOT UPDATED")
            return {"success": True, "message": f"Slot successfully updated for {doctor_name}."}
        
        return {"success": False, "message": "Appointment slot not found in database."}
    except Exception as e:
        db.rollback()
        return {"success": False, "message": f"Database error: {str(e)}"}
    finally:
        db.close()