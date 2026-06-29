from src.database import db

if __name__ == "__main__":
    print("Initiating System Check...")
    success = db.verify_connections()

    if success:
        print("\nAll systems nominal. Ready for data ingestion.")
    db.close()
