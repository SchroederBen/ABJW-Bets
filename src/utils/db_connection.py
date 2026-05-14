from sqlalchemy import create_engine
from src.utils.config import DATABASE_URL

# Engine for connecting to the PostgreSQL database using SQLAlchemy
# makes things slightly simpler to manage connections and execute queries
def get_engine():
    # Returns a SQLAlchemy engine instance for Neon/PostgreSQL connection
    return create_engine(DATABASE_URL)