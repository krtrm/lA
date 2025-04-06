import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta
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
            
            # Blogs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS blogs (
                blog_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                category TEXT,
                tags TEXT,
                is_official BOOLEAN DEFAULT FALSE,
                source_id INTEGER,
                source_url TEXT,
                published BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                published_at TIMESTAMP,
                likes INTEGER DEFAULT 0,
                views INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
            
            # Blog comments table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS blog_comments (
                comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                blog_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (blog_id) REFERENCES blogs(blog_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
            
            # News sources table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT,
                logo_url TEXT,
                description TEXT,
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    # Blog Management
    def create_blog(self, user_id: str, title: str, content: str, 
                   summary: str = None, category: str = None,
                   tags: List[str] = None, is_official: bool = False,
                   source_id: int = None, source_url: str = None) -> Optional[Dict]:
        """Create a new blog post"""
        conn = self.get_connection()
        try:
            now = datetime.now().isoformat()
            tags_json = json.dumps(tags) if tags else None
            
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO blogs (
                    user_id, title, content, summary, category, 
                    tags, is_official, source_id, source_url,
                    created_at, updated_at, published_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, title, content, summary, category, 
                 tags_json, is_official, source_id, source_url,
                 now, now, now)
            )
            blog_id = cursor.lastrowid
            conn.commit()
            
            # Retrieve the new blog
            cursor.execute("SELECT * FROM blogs WHERE blog_id = ?", (blog_id,))
            blog = cursor.fetchone()
            
            # Format the blog data
            if blog and blog.get("tags"):
                blog["tags"] = json.loads(blog["tags"])
                
            return blog
        except Exception as e:
            logger.error(f"Error creating blog: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_blogs(self, limit: int = 10, offset: int = 0, 
                 category: str = None, user_id: str = None,
                 is_official: bool = None, published_only: bool = True) -> List[Dict]:
        """Get blogs with optional filtering"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT b.*, 
                       u.first_name, u.last_name,
                       (SELECT COUNT(*) FROM blog_comments WHERE blog_id = b.blog_id) as comment_count,
                       ns.name as source_name, ns.logo_url as source_logo
                FROM blogs b
                JOIN users u ON b.user_id = u.user_id
                LEFT JOIN news_sources ns ON b.source_id = ns.source_id
                WHERE 1=1
            """
            params = []
            
            if category:
                query += " AND b.category = ?"
                params.append(category)
                
            if user_id:
                query += " AND b.user_id = ?"
                params.append(user_id)
                
            if is_official is not None:
                query += " AND b.is_official = ?"
                params.append(1 if is_official else 0)
                
            if published_only:
                query += " AND b.published = 1"
                
            query += " ORDER BY b.created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            blogs = cursor.fetchall()
            
            # Format blog data
            for blog in blogs:
                if blog.get("tags"):
                    blog["tags"] = json.loads(blog["tags"])
                    
                # Format dates
                for date_field in ["created_at", "updated_at", "published_at"]:
                    if blog.get(date_field):
                        blog[date_field] = datetime.fromisoformat(blog[date_field]).strftime("%Y-%m-%d")
                
                # Calculate read time (approx. 200 words per minute)
                word_count = len(blog.get("content", "").split())
                blog["read_time_minutes"] = max(1, round(word_count / 200))
                
                # Create author display name
                blog["author"] = f"{blog.get('first_name', '')} {blog.get('last_name', '')}"
            
            return blogs
        except Exception as e:
            logger.error(f"Error getting blogs: {e}")
            return []
        finally:
            conn.close()
    
    def get_blog(self, blog_id: int) -> Optional[Dict]:
        """Get a blog by ID"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT b.*, 
                       u.first_name, u.last_name,
                       (SELECT COUNT(*) FROM blog_comments WHERE blog_id = b.blog_id) as comment_count,
                       ns.name as source_name, ns.logo_url as source_logo
                FROM blogs b
                JOIN users u ON b.user_id = u.user_id
                LEFT JOIN news_sources ns ON b.source_id = ns.source_id
                WHERE b.blog_id = ?
            """, (blog_id,))
            
            blog = cursor.fetchone()
            
            if not blog:
                return None
                
            # Format blog data
            if blog.get("tags"):
                blog["tags"] = json.loads(blog["tags"])
                
            # Format dates
            for date_field in ["created_at", "updated_at", "published_at"]:
                if blog.get(date_field):
                    blog[date_field] = datetime.fromisoformat(blog[date_field]).strftime("%Y-%m-%d")
            
            # Calculate read time (approx. 200 words per minute)
            word_count = len(blog.get("content", "").split())
            blog["read_time_minutes"] = max(1, round(word_count / 200))
            
            # Create author display name
            blog["author"] = f"{blog.get('first_name', '')} {blog.get('last_name', '')}"
            
            # Increment view count
            cursor.execute(
                "UPDATE blogs SET views = views + 1 WHERE blog_id = ?",
                (blog_id,)
            )
            conn.commit()
            
            return blog
        except Exception as e:
            logger.error(f"Error getting blog: {e}")
            return None
        finally:
            conn.close()
    
    def update_blog(self, blog_id: int, user_id: str, updates: Dict) -> Optional[Dict]:
        """Update a blog post"""
        conn = self.get_connection()
        try:
            # First verify the blog exists and belongs to the user
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM blogs WHERE blog_id = ? AND user_id = ?",
                (blog_id, user_id)
            )
            blog = cursor.fetchone()
            
            if not blog:
                return None
                
            # Prepare update fields
            update_fields = []
            params = []
            
            for field in ["title", "content", "summary", "category", "published"]:
                if field in updates:
                    update_fields.append(f"{field} = ?")
                    params.append(updates[field])
            
            # Handle tags separately (convert to JSON)
            if "tags" in updates:
                update_fields.append("tags = ?")
                params.append(json.dumps(updates["tags"]))
            
            # Always update the updated_at field
            now = datetime.now().isoformat()
            update_fields.append("updated_at = ?")
            params.append(now)
            
            # If publishing status changed to true, update published_at
            if updates.get("published") == True and blog["published"] == 0:
                update_fields.append("published_at = ?")
                params.append(now)
            
            # Perform the update
            params.append(blog_id)
            cursor.execute(
                f"UPDATE blogs SET {', '.join(update_fields)} WHERE blog_id = ?",
                params
            )
            conn.commit()
            
            # Return the updated blog
            return self.get_blog(blog_id)
        except Exception as e:
            logger.error(f"Error updating blog: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def delete_blog(self, blog_id: int, user_id: str) -> bool:
        """Delete a blog post"""
        conn = self.get_connection()
        try:
            # First verify the blog exists and belongs to the user
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM blogs WHERE blog_id = ? AND user_id = ?",
                (blog_id, user_id)
            )
            blog = cursor.fetchone()
            
            if not blog:
                return False
                
            # Delete all comments first
            cursor.execute(
                "DELETE FROM blog_comments WHERE blog_id = ?",
                (blog_id,)
            )
            
            # Then delete the blog
            cursor.execute(
                "DELETE FROM blogs WHERE blog_id = ?",
                (blog_id,)
            )
            conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting blog: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def add_blog_comment(self, blog_id: int, user_id: str, content: str) -> Optional[Dict]:
        """Add a comment to a blog"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO blog_comments (blog_id, user_id, content) VALUES (?, ?, ?)",
                (blog_id, user_id, content)
            )
            comment_id = cursor.lastrowid
            conn.commit()
            
            # Return the new comment
            cursor.execute("""
                SELECT c.*, u.first_name, u.last_name 
                FROM blog_comments c
                JOIN users u ON c.user_id = u.user_id
                WHERE c.comment_id = ?
            """, (comment_id,))
            comment = cursor.fetchone()
            
            if comment:
                # Format timestamp
                comment["created_at"] = datetime.fromisoformat(comment["created_at"]).strftime("%Y-%m-%d %H:%M")
                # Add author name
                comment["author"] = f"{comment.get('first_name', '')} {comment.get('last_name', '')}"
            
            return comment
        except Exception as e:
            logger.error(f"Error adding blog comment: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_blog_comments(self, blog_id: int) -> List[Dict]:
        """Get all comments for a blog"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, u.first_name, u.last_name 
                FROM blog_comments c
                JOIN users u ON c.user_id = u.user_id
                WHERE c.blog_id = ?
                ORDER BY c.created_at DESC
            """, (blog_id,))
            comments = cursor.fetchall()
            
            # Format comments
            for comment in comments:
                # Format timestamp
                comment["created_at"] = datetime.fromisoformat(comment["created_at"]).strftime("%Y-%m-%d %H:%M")
                # Add author name
                comment["author"] = f"{comment.get('first_name', '')} {comment.get('last_name', '')}"
            
            return comments
        except Exception as e:
            logger.error(f"Error getting blog comments: {e}")
            return []
        finally:
            conn.close()
    
    def like_blog(self, blog_id: int) -> bool:
        """Increment the like count for a blog"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE blogs SET likes = likes + 1 WHERE blog_id = ?",
                (blog_id,)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error liking blog: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_blog_categories(self) -> List[str]:
        """Get all unique blog categories"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT category FROM blogs WHERE category IS NOT NULL"
            )
            categories = cursor.fetchall()
            return [category["category"] for category in categories]
        except Exception as e:
            logger.error(f"Error getting blog categories: {e}")
            return []
        finally:
            conn.close()
    
    # News Sources Management
    def add_news_source(self, name: str, url: str = None, 
                        logo_url: str = None, description: str = None,
                        is_verified: bool = False) -> Optional[Dict]:
        """Add a new news source"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO news_sources (name, url, logo_url, description, is_verified)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, url, logo_url, description, is_verified)
            )
            source_id = cursor.lastrowid
            conn.commit()
            
            # Return the new source
            cursor.execute("SELECT * FROM news_sources WHERE source_id = ?", (source_id,))
            source = cursor.fetchone()
            
            return source
        except Exception as e:
            logger.error(f"Error adding news source: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_news_sources(self, verified_only: bool = False) -> List[Dict]:
        """Get all news sources"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = "SELECT * FROM news_sources"
            params = []
            
            if verified_only:
                query += " WHERE is_verified = 1"
                
            query += " ORDER BY name ASC"
            
            cursor.execute(query, params)
            sources = cursor.fetchall()
            
            return sources
        except Exception as e:
            logger.error(f"Error getting news sources: {e}")
            return []
        finally:
            conn.close()
    
    def add_official_news(self, title: str, content: str, summary: str,
                         source_id: int, source_url: str, user_id: str,
                         category: str = None, tags: List[str] = None) -> Optional[Dict]:
        """Add official news article"""
        return self.create_blog(
            user_id=user_id,
            title=title,
            content=content,
            summary=summary,
            category=category,
            tags=tags,
            is_official=True,
            source_id=source_id,
            source_url=source_url
        )

# Create a singleton database instance
db = Database()
