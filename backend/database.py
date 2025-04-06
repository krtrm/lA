import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vaqeel.db")

def dict_factory(cursor, row):
    """Convert SQLite row to dictionary"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class Database:
    def __init__(self, db_path=DB_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        self.initialize_db()
    
    def get_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = dict_factory
        return conn
    
    def initialize_db(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT,
                first_name TEXT,
                last_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                preferences TEXT
            )
            ''')
            
            # Spaces table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS spaces (
                space_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
            
            # Messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (space_id) REFERENCES spaces(space_id)
            )
            ''')
            
            # Usage stats table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    # User Management
    def get_or_create_user(self, user_id: str, email: str, first_name: str, last_name: str) -> Dict:
        """Get user by ID or create if doesn't exist"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                # Create new user
                now = datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO users (user_id, email, first_name, last_name, last_login) VALUES (?, ?, ?, ?, ?)",
                    (user_id, email, first_name, last_name, now)
                )
                conn.commit()
                
                # Retrieve newly created user
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                user = cursor.fetchone()
            else:
                # Update last login
                now = datetime.now().isoformat()
                cursor.execute(
                    "UPDATE users SET last_login = ? WHERE user_id = ?",
                    (now, user_id)
                )
                conn.commit()
            
            return user
        except Exception as e:
            logger.error(f"Error in get_or_create_user: {e}")
            conn.rollback()
            return {}
        finally:
            conn.close()
    
    def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences"""
        conn = self.get_connection()
        try:
            prefs_json = json.dumps(preferences)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET preferences = ? WHERE user_id = ?",
                (prefs_json, user_id)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    # Spaces Management
    def create_space(self, user_id: str, title: str, space_type: str) -> Optional[Dict]:
        """Create a new space for a user"""
        conn = self.get_connection()
        try:
            now = datetime.now().isoformat()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO spaces (user_id, title, type, last_active) VALUES (?, ?, ?, ?)",
                (user_id, title, space_type, now)
            )
            space_id = cursor.lastrowid
            conn.commit()
            
            # Retrieve the new space
            cursor.execute("SELECT * FROM spaces WHERE space_id = ?", (space_id,))
            space = cursor.fetchone()
            return space
        except Exception as e:
            logger.error(f"Error creating space: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_user_spaces(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict]:
        """Get spaces for a user"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, 
                       (SELECT COUNT(*) FROM messages WHERE space_id = s.space_id) as message_count,
                       (SELECT content FROM messages WHERE space_id = s.space_id ORDER BY timestamp DESC LIMIT 1) as last_message
                FROM spaces s 
                WHERE s.user_id = ? 
                ORDER BY s.last_active DESC 
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset)
            )
            spaces = cursor.fetchall()
            
            # Format datetime strings
            for space in spaces:
                space["created_at"] = datetime.fromisoformat(space["created_at"]).strftime("%Y-%m-%d")
                if space["last_active"]:
                    space["last_active"] = datetime.fromisoformat(space["last_active"]).strftime("%Y-%m-%d")
            
            return spaces
        except Exception as e:
            logger.error(f"Error getting user spaces: {e}")
            return []
        finally:
            conn.close()
    
    def get_space(self, space_id: int) -> Optional[Dict]:
        """Get a space by ID"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM spaces WHERE space_id = ?", (space_id,))
            space = cursor.fetchone()
            
            if space:
                space["created_at"] = datetime.fromisoformat(space["created_at"]).strftime("%Y-%m-%d")
                if space["last_active"]:
                    space["last_active"] = datetime.fromisoformat(space["last_active"]).strftime("%Y-%m-%d")
            
            return space
        except Exception as e:
            logger.error(f"Error getting space: {e}")
            return None
        finally:
            conn.close()
    
    def update_space_activity(self, space_id: int) -> bool:
        """Update the last active timestamp of a space"""
        conn = self.get_connection()
        try:
            now = datetime.now().isoformat()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE spaces SET last_active = ? WHERE space_id = ?",
                (now, space_id)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating space activity: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    # Messages Management
    def add_message(self, space_id: int, role: str, content: str, metadata: Dict = None) -> Optional[Dict]:
        """Add a message to a space"""
        conn = self.get_connection()
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (space_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (space_id, role, content, metadata_json)
            )
            message_id = cursor.lastrowid
            conn.commit()
            
            # Update space activity timestamp
            self.update_space_activity(space_id)
            
            # Return the new message
            cursor.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,))
            message = cursor.fetchone()
            
            if message:
                # Format timestamp
                message["timestamp"] = datetime.fromisoformat(message["timestamp"]).strftime("%H:%M")
                # Parse metadata JSON if exists
                if message["metadata"]:
                    message["metadata"] = json.loads(message["metadata"])
            
            return message
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_space_messages(self, space_id: int) -> List[Dict]:
        """Get all messages for a space"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE space_id = ? ORDER BY timestamp ASC",
                (space_id,)
            )
            messages = cursor.fetchall()
            
            # Format messages
            for message in messages:
                # Format timestamp
                message["timestamp"] = datetime.fromisoformat(message["timestamp"]).strftime("%H:%M")
                # Parse metadata JSON if exists
                if message["metadata"]:
                    message["metadata"] = json.loads(message["metadata"])
            
            return messages
        except Exception as e:
            logger.error(f"Error getting space messages: {e}")
            return []
        finally:
            conn.close()
    
    # Usage Statistics
    def log_user_action(self, user_id: str, action_type: str, details: Dict = None) -> bool:
        """Log a user action for analytics"""
        conn = self.get_connection()
        try:
            details_json = json.dumps(details) if details else None
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO usage_stats (user_id, action_type, details) VALUES (?, ?, ?)",
                (user_id, action_type, details_json)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging user action: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Get total spaces
            cursor.execute("SELECT COUNT(*) as total_spaces FROM spaces WHERE user_id = ?", (user_id,))
            spaces_result = cursor.fetchone()
            total_spaces = spaces_result["total_spaces"] if spaces_result else 0
            
            # Get total messages
            cursor.execute("""
                SELECT COUNT(*) as total_messages 
                FROM messages m 
                JOIN spaces s ON m.space_id = s.space_id 
                WHERE s.user_id = ?
            """, (user_id,))
            messages_result = cursor.fetchone()
            total_messages = messages_result["total_messages"] if messages_result else 0
            
            # Get messages this month
            current_month = datetime.now().strftime("%Y-%m")
            cursor.execute("""
                SELECT COUNT(*) as monthly_messages 
                FROM messages m 
                JOIN spaces s ON m.space_id = s.space_id 
                WHERE s.user_id = ? AND strftime('%Y-%m', m.timestamp) = ?
            """, (user_id, current_month))
            monthly_result = cursor.fetchone()
            monthly_messages = monthly_result["monthly_messages"] if monthly_result else 0
            
            # Get active researches (spaces with activity in last 7 days)
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) as active_researches 
                FROM spaces 
                WHERE user_id = ? AND last_active > ?
            """, (user_id, seven_days_ago))
            active_result = cursor.fetchone()
            active_researches = active_result["active_researches"] if active_result else 0
            
            return {
                "total_spaces": total_spaces,
                "total_messages": total_messages,
                "messages_this_month": monthly_messages,
                "active_researches": active_researches
            }
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {
                "total_spaces": 0,
                "total_messages": 0,
                "messages_this_month": 0,
                "active_researches": 0
            }
        finally:
            conn.close()

# Create a singleton database instance
db = Database()
