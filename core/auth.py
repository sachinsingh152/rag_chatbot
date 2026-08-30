import sqlite3
import bcrypt
import os
import json

DATA_DIR = os.getenv('DATA_DIR', '.')
DB_PATH = os.path.join(DATA_DIR, 'users.db')

def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations TEXT,
            workflow TEXT,
            conversation_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Run migration to add conversation_id if it doesn't exist
    try:
        cursor.execute('ALTER TABLE chat_history ADD COLUMN conversation_id TEXT')
        # Assign a default conversation id for existing messages
        import uuid
        cursor.execute('UPDATE chat_history SET conversation_id = ? WHERE conversation_id IS NULL', (str(uuid.uuid4()),))
    except sqlite3.OperationalError:
        pass # Column already exists
        
    conn.commit()
    conn.close()


def create_user(username, password):
    """
    Creates a new user with a hashed password.
    Returns True if successful, False if the username already exists.
    """
    # Hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, hashed_password.decode('utf-8'))
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        # Username already exists
        success = False
    finally:
        conn.close()
    
    return success

def authenticate_user(username, password):
    """
    Authenticates a user by comparing the provided password with the stored hash.
    Returns True if authentication is successful, False otherwise.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        stored_hash = result[0].encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True
    
    return False

def save_chat_message(username, conversation_id, message_dict):
    """Saves a chat message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    citations_json = json.dumps(message_dict.get('citations', []))
    workflow_json = json.dumps(message_dict.get('workflow', []))
    
    cursor.execute(
        'INSERT INTO chat_history (username, conversation_id, role, content, citations, workflow) VALUES (?, ?, ?, ?, ?, ?)',
        (username, conversation_id, message_dict['role'], message_dict['content'], citations_json, workflow_json)
    )
    conn.commit()
    conn.close()

def load_chat_history(username, conversation_id):
    """Loads the chat history for a specific user and conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT role, content, citations, workflow FROM chat_history WHERE username = ? AND conversation_id = ? ORDER BY timestamp ASC',
        (username, conversation_id)
    )
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        role, content, citations_str, workflow_str = row
        messages.append({
            'role': role,
            'content': content,
            'citations': json.loads(citations_str) if citations_str else [],
            'workflow': json.loads(workflow_str) if workflow_str else []
        })
    
    return messages

def get_user_conversations(username):
    """Retrieves all conversations for a user, sorted by most recent."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get distinct conversation IDs and the first user message as the title, along with max timestamp for sorting
    query = '''
        SELECT c1.conversation_id, c1.content, MAX(c2.timestamp) as last_updated
        FROM chat_history c1
        JOIN chat_history c2 ON c1.conversation_id = c2.conversation_id
        WHERE c1.username = ? AND c1.role = 'user'
        GROUP BY c1.conversation_id
        ORDER BY last_updated DESC
    '''
    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    conn.close()
    
    conversations = []
    for row in rows:
        conv_id, first_msg, _ = row
        # Use first 30 chars of the message as title
        title = first_msg[:30] + '...' if len(first_msg) > 30 else first_msg
        conversations.append({
            'id': conv_id,
            'title': title
        })
        
    return conversations

def clear_chat_history(username, conversation_id=None):
    """Deletes chat messages. If conversation_id is None, clears all for the user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if conversation_id:
        cursor.execute('DELETE FROM chat_history WHERE username = ? AND conversation_id = ?', (username, conversation_id))
    else:
        cursor.execute('DELETE FROM chat_history WHERE username = ?', (username,))
    conn.commit()
    conn.close()
