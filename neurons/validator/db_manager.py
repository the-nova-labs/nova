import sqlite3
import os

class DBManager:
    def __init__(self):
        """
        Uses 'neurons/validator/miner_best_scores.db' as the SQLite file.
        Creates the directory if needed.
        """
        self.db_path = os.path.join('neurons', 'validator', 'miner_best_scores.db')
        # Ensure the directory for the DB exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        """
        Creates the table if it doesn't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Schema for storing each miner's best score
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS miner_bests (
                miner_uid INTEGER PRIMARY KEY,
                best_smiles TEXT,
                best_score REAL
            )
        """)

        conn.commit()
        conn.close()

    def update_best_score(self, miner_uid: int, smiles: str, score: float):
        """
        Inserts or updates a single miner's best smiles and score.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fetch current record
        cursor.execute("SELECT best_score FROM miner_bests WHERE miner_uid = ?", (miner_uid,))
        row = cursor.fetchone()

        if row is None:
            # No entry yet, insert
            cursor.execute("""
                INSERT INTO miner_bests (miner_uid, best_smiles, best_score)
                VALUES (?, ?, ?)
            """, (miner_uid, smiles, score))
        else:
            current_best_score = row[0]
            # Only update if new score is higher
            if score > current_best_score:
                cursor.execute("""
                    UPDATE miner_bests 
                    SET best_smiles = ?, best_score = ?
                    WHERE miner_uid = ?
                """, (smiles, score, miner_uid))

        conn.commit()
        conn.close()

    def get_best_score(self, miner_uid: int):
        """
        Returns (best_smiles, best_score) for a miner, or (None, 0) if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT best_smiles, best_score FROM miner_bests WHERE miner_uid = ?", (miner_uid,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return row[0], row[1]
        else:
            return None, 0

