import os
from dotenv import load_dotenv
import pyotp
import sys

# Add the parent directory to sys.path to allow importing from other modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_totp_token():
    """
    Loads the TOTP secret from the environment variable BROKER_TOTP_KEY
    and generates the current TOTP token.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Read the TOTP secret key
    secret = os.getenv("BROKER_TOTP_KEY")

    if not secret:
        print("Error: BROKER_TOTP_KEY not found in environment or .env file.")
        return None

    try:
        # Create TOTP object
        totp = pyotp.TOTP(secret.strip().replace(" ", ""))
        # Generate current token
        return totp.now()
    except Exception as e:
        print(f"Error generating TOTP: {e}")
        return None

if __name__ == "__main__":
    token = get_totp_token()
    if token:
        print(f"Current TOTP Token: {token}")
