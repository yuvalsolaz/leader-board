
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'robin',
    'user': 'postgres',
    'password': 'robin',
    'table_name': 'experiments'
}

def get_connection_string():
    return f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"