from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import User, get_db
from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthHandler:
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    async def authenticate_user(self, email: str, password: str, db: AsyncSession) -> Optional[User]:
        result = await db.execute(select(User).filter(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    async def get_current_user(self, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            email: str = payload.get("sub")
            if email is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception

        result = await db.execute(select(User).filter(User.email == email))
        user = result.scalar_one_or_none()
        if user is None:
            raise credentials_exception
        return user

    def require_auth(self, func):
        """Decorator for NiceGUI pages requiring authentication"""
        async def wrapper(*args, **kwargs):
            # Implementation of session-based auth check for NiceGUI
            # This would integrate with your session management
            if not self.is_authenticated():
                return self.redirect_to_login()
            return await func(*args, **kwargs)
        return wrapper

    def is_authenticated(self) -> bool:
        # Implement session-based authentication check
        # This would integrate with your session management
        return True  # Placeholder

    def redirect_to_login(self):
        # Implement redirect to login page
        pass

auth_handler = AuthHandler() 