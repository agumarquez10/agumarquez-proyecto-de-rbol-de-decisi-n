from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import pandas as pd

def load_dataset(path="data/diabetes.csv"):
    return pd.read_csv(path)

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine
