import os
import sys
from contextlib import contextmanager
from pathlib import Path
import psycopg
from psycopg.rows import dict_row


@contextmanager
def connect():
    with psycopg.connect(os.environ['SA_DATABASE_URL'], row_factory=dict_row,
                         connect_timeout=5, options='-c statement_timeout=15000') as conn:
        yield conn


def migrate():
    path = Path(__file__).resolve().parents[2] / 'migrations'
    if not path.is_dir():
        path = Path(sys.prefix) / 'share/sales-brain-public/migrations'
    if not list(path.glob('*.sql')):
        raise RuntimeError('sales_brain_migration_files_missing')
    with connect() as conn:
        conn.execute('SELECT pg_advisory_xact_lock(%s)', (7310101,))
        conn.execute('CREATE TABLE IF NOT EXISTS sb_migrations(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())')
        for file in sorted(path.glob('*.sql')):
            if conn.execute('SELECT 1 FROM sb_migrations WHERE version=%s', (file.name,)).fetchone():
                continue
            conn.execute(file.read_text())
            conn.execute('INSERT INTO sb_migrations(version) VALUES(%s)', (file.name,))
    from .agents import seed_defaults
    seed_defaults(os.environ.get('SA_TENANT_ID','example-tenant'))


def main():
    migrate()
    print('sales-brain migrations and region defaults applied')


if __name__ == '__main__':
    main()
