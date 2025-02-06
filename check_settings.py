from app import *

def check_settings():
    with db_session() as session:
        settings = session.query(Settings).all()
        print("\nCurrent Settings in Database:")
        print("============================")
        for s in settings:
            print(f"\nType: {s.setting_type}")
            print(f"Name: {s.name}")
            print(f"Value: {s.value}")
            print(f"Active: {s.is_active}")
            print("----------------------------")

if __name__ == "__main__":
    check_settings() 