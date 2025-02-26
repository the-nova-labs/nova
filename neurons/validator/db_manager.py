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
                miner_hotkey TEXT,
                best_smiles TEXT,
                best_score REAL,
                bounty_points INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def clear_db(self):
        """
        Removes all rows from the miner_bests table
        Effectively resets the table for reuse in the next challenge
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM miner_bests")
        conn.commit()
        conn.close()

    def update_best_score(self, miner_uid: int, miner_hotkey: str, smiles: str, score: float):
        """
        Inserts or updates a single miner's best smiles and score.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        miner_uid = int(miner_uid)
        score = float(score)

        # Check if there's an existing record for this miner_uid
        cursor.execute("SELECT best_score FROM miner_bests WHERE miner_uid = ?", (miner_uid,))
        row = cursor.fetchone()

        if row is None:
            # No entry yet, so insert a new record
            cursor.execute("""
                INSERT INTO miner_bests (miner_uid, miner_hotkey, best_smiles, best_score)
                VALUES (?, ?, ?, ?)
            """, (miner_uid, miner_hotkey, smiles, score))
        else:
            current_best_score = row[0]
            # Only update if the new score is higher than the existing one
            if score > current_best_score:
                cursor.execute("""
                    UPDATE miner_bests
                    SET miner_hotkey = ?, best_smiles = ?, best_score = ?
                    WHERE miner_uid = ?
                """, (miner_hotkey, smiles, score, miner_uid))

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
        
    def get_all_best_scores(self):
        """
        Retrieves all miners' best scores from the database.
        Returns a list of tuples: (miner_uid, miner_hotkey, best_smiles, best_score, bounty_points)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT miner_uid, miner_hotkey, best_smiles, best_score, bounty_points
            FROM miner_bests
        """)

        rows = cursor.fetchall()
        conn.close()
    
        return rows
        
    def award_bounty(self, bounty_value: int):
        """
        Finds the row with the highest best_score and adds `bounty_value` to that miner's bounty_points.
        If the table is empty, or all best_score entries are NULL, does nothing.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find the miner with the highest best_score
        cursor.execute("""
            SELECT miner_uid, bounty_points
            FROM miner_bests
            ORDER BY best_score DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        # If there is at least one row, award bounty to that miner
        if row is not None:
            miner_uid, current_bounty_points = row

            # If current_bounty_points is None, treat as 0
            if current_bounty_points is None:
                current_bounty_points = 0

            new_bounty_points = current_bounty_points + bounty_value

            cursor.execute("""
                UPDATE miner_bests
                SET bounty_points = ?
                WHERE miner_uid = ?
            """, (new_bounty_points, miner_uid))

        conn.commit()
        conn.close()
