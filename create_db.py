import sqlite3
import pandas as pd
from pathlib import Path

db_path = 'data/tyres.db'
# Path(db_path).touch()
conn = sqlite3.connect(db_path)
c = conn.cursor()
# query = '''CREATE TABLE pricelist (SKU text, Brand text, Model text, Price int)'''
# c.execute(query)

# Create pricelist table
pricelist = pd.read_csv('data/pricelist.csv')
pricelist.to_sql('pricelist', conn, if_exists='replace', index=False)
c.execute('''SELECT * FROM pricelist''').fetchall() 

# Create inventory table
inventory = pd.read_csv('data/inventory.csv')
inventory.to_sql('inventory', conn, if_exists='replace', index=False)
c.execute('''SELECT * FROM inventory''').fetchall() 

# List all tables in tyres database
c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
# List columns in inventory table
tab_info = c.execute(f"PRAGMA table_info('inventory')").fetchall()
