import psycopg2
import os
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from datetime import date
from utils import AnsiColors

load_dotenv()

def get_connection():
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=os.getenv('POSTGRES_PORT'),
            dbname=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id            SERIAL PRIMARY KEY,
                    name          VARCHAR(255) UNIQUE NOT NULL,
                    registered_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS attendance (
                    id           SERIAL PRIMARY KEY,
                    student_id   INTEGER NOT NULL REFERENCES students(id),
                    attended_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    session_date DATE NOT NULL,
                    UNIQUE (student_id, session_date)
                );

                CREATE INDEX IF NOT EXISTS idx_attendance_date
                    ON attendance (session_date);
            """)
            cur.execute("ALTER DATABASE attendance_system_db SET timezone TO 'America/Sao_Paulo';")
        conn.commit()
    print(f"{AnsiColors.GREEN}✔ Banco de dados inicializado com sucesso.{AnsiColors.RESET}")


def upsert_student(name: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO students (name) VALUES (%s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id;
            """, (name,))
            row = cur.fetchone()
        conn.commit()
    return row[0]


def record_attendance(student_name: str, session_date: date):
    student_id = upsert_student(student_name)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT attended_at FROM attendance
                WHERE student_id = %s AND session_date = %s;
            """, (student_id, session_date))
            existing = cur.fetchone()
            if existing:
                return False, existing[0]
            cur.execute("""
                INSERT INTO attendance (student_id, session_date, attended_at)
                VALUES (%s, %s, NOW()) RETURNING attended_at;
            """, (student_id, session_date))
            ts = cur.fetchone()[0]
        conn.commit()
    return True, ts


def get_attendance_by_date(session_date: date):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT s.name, a.attended_at
                FROM attendance a
                JOIN students s ON s.id = a.student_id
                WHERE a.session_date = %s
                ORDER BY a.attended_at;
            """, (session_date,))
            return cur.fetchall()


def get_all_session_dates():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT session_date FROM attendance
                ORDER BY session_date DESC;
            """)
            return [r[0] for r in cur.fetchall()]


def get_attendance_summary():
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT s.name,
                       COUNT(a.id)       AS total_days,
                       MAX(a.session_date) AS last_seen
                FROM students s
                LEFT JOIN attendance a ON a.student_id = s.id
                GROUP BY s.name
                ORDER BY total_days DESC, s.name;
            """)
            return cur.fetchall()