from datetime import timedelta
from passlib.context import CryptContext


passwordContext = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# function to chek
def verifyPassword(plainPassword:str,hashedPassword:str):
    return passwordContext.verify(plainPassword,hashedPassword)

