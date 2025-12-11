"""
Wikipedia Event Stream Processor
Monitors Wikipedia events for selected IMDB entities
"""

import json
import sqlite3
import time
from datetime import datetime
from collections import defaultdict, deque
import requests
from sseclient import SSEClient
import threading

class WikipediaStreamProcessor:
    """
    Processes Wikipedia event streams for selected entities
    Tracks metrics and generates alerts
    """
    
    def __init__(self, entities, db_path='../outputs/metrics.db', alert_path='../outputs/alerts.json'):
        """
        Initialize the stream processor
        
        Args:
            entities (list): List of entity names to monitor
            db_path (str): Path to SQLite database for metrics
            alert_path (str): Path to JSON file for alerts
        """
        self.entities = entities
        self.db_path = db_path
        self.alert_path = alert_path
        
        # Metrics storage
        self.metrics = defaultdict(lambda: {
            'edit_count': 0,
            'total_bytes_changed': 0,
            'unique_users': set(),
            'anonymous_edits': 0,
            'bot_edits': 0,
            'last_edit_time': None,
            'edit_timestamps': deque(maxlen=100)  # Keep last 100 edit timestamps
        })
        
        # Alert configuration
        self.alert_threshold = 5  # Alert if more than 5 edits per hour
        self.alerts = []
        
        # Initialize database
        self.init_database()
        
        print(f"Stream processor initialized for {len(entities)} entities")
        print(f"Entities: {', '.join(entities)}")
        
    def init_database(self):
        """Initialize SQLite database for storing metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                edit_count INTEGER,
                bytes_changed INTEGER,
                unique_users INTEGER,
                anonymous_edits INTEGER,
                bot_edits INTEGER
            )
        ''')
        
        # Create events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT NOT NULL,
                timestamp DATETIME,
                user TEXT,
                is_bot BOOLEAN,
                is_anonymous BOOLEAN,
                bytes_changed INTEGER,
                comment TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")
    
    def save_metrics(self):
        """Save current metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for entity, metrics in self.metrics.items():
            cursor.execute('''
                INSERT INTO entity_metrics 
                (entity_name, edit_count, bytes_changed, unique_users, anonymous_edits, bot_edits)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entity,
                metrics['edit_count'],
                metrics['total_bytes_changed'],
                len(metrics['unique_users']),
                metrics['anonymous_edits'],
                metrics['bot_edits']
            ))
        
        conn.commit()
        conn.close()
    
    def save_event(self, entity, event_data):
        """Save individual edit event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO edit_events 
            (entity_name, timestamp, user, is_bot, is_anonymous, bytes_changed, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            entity,
            event_data.get('timestamp'),
            event_data.get('user'),
            event_data.get('bot', False),
            event_data.get('anon', False),
            event_data.get('length', {}).get('new', 0) - event_data.get('length', {}).get('old', 0),
            event_data.get('comment', '')
        ))
        
        conn.commit()
        conn.close()
    
    def check_alert_condition(self, entity):
        """
        Check if alert condition is met for an entity
        Alert if more than threshold edits in the last hour
        """
        metrics = self.metrics[entity]
        timestamps = metrics['edit_timestamps']
        
        if len(timestamps) < 2:
            return False
        
        # Check edits in last hour
        current_time = time.time()
        recent_edits = sum(1 for ts in timestamps if current_time - ts < 3600)
        
        return recent_edits > self.alert_threshold
    
    def generate_alert(self, entity, reason, data):
        """Generate and save an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'entity': entity,
            'reason': reason,
            'data': data
        }
        
        self.alerts.append(alert)
        
        # Save to file
        try:
            with open(self.alert_path, 'a') as f:
                f.write(json.dumps(alert) + '\n')
            
            print(f"\n{'='*60}")
            print(f"🚨 ALERT: {entity}")
            print(f"Reason: {reason}")
            print(f"Details: {json.dumps(data, indent=2)}")
            print('='*60 + '\n')
        except Exception as e:
            print(f"Error saving alert: {e}")
    
    def process_event(self, event_data):
        """Process a single Wikipedia event"""
        try:
            # Check if this event is for one of our monitored entities
            title = event_data.get('title', '')
            
            # Check if title matches any of our entities
            for entity in self.entities:
                if entity.lower() in title.lower() or title.lower() in entity.lower():
                    # Update metrics
                    metrics = self.metrics[entity]
                    metrics['edit_count'] += 1
                    metrics['last_edit_time'] = datetime.now()
                    metrics['edit_timestamps'].append(time.time())
                    
                    # Calculate bytes changed
                    if 'length' in event_data:
                        old_length = event_data['length'].get('old', 0)
                        new_length = event_data['length'].get('new', 0)
                        bytes_changed = new_length - old_length
                        metrics['total_bytes_changed'] += abs(bytes_changed)
                    
                    # Track user
                    user = event_data.get('user', 'Unknown')
                    metrics['unique_users'].add(user)
                    
                    # Check for anonymous edit
                    if event_data.get('anon', False):
                        metrics['anonymous_edits'] += 1
                    
                    # Check for bot edit
                    if event_data.get('bot', False):
                        metrics['bot_edits'] += 1
                    
                    # Save event to database
                    self.save_event(entity, event_data)
                    
                    # Check alert condition
                    if self.check_alert_condition(entity):
                        self.generate_alert(
                            entity,
                            'High edit frequency detected',
                            {
                                'edits_last_hour': sum(
                                    1 for ts in metrics['edit_timestamps'] 
                                    if time.time() - ts < 3600
                                ),
                                'threshold': self.alert_threshold,
                                'last_editor': user
                            }
                        )
                    
                    # Check for anonymous edit alert
                    if event_data.get('anon', False):
                        self.generate_alert(
                            entity,
                            'Anonymous edit detected',
                            {
                                'user': user,
                                'comment': event_data.get('comment', 'No comment'),
                                'timestamp': event_data.get('timestamp')
                            }
                        )
                    
                    # Print progress
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"{entity}: Edit #{metrics['edit_count']} by {user}")
                    
                    break
        
        except Exception as e:
            print(f"Error processing event: {e}")
    
    def print_metrics_summary(self):
        """Print current metrics summary"""
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        
        for entity, metrics in self.metrics.items():
            print(f"\n{entity}:")
            print(f"  Total edits: {metrics['edit_count']}")
            print(f"  Total bytes changed: {metrics['total_bytes_changed']:,}")
            print(f"  Unique users: {len(metrics['unique_users'])}")
            print(f"  Anonymous edits: {metrics['anonymous_edits']}")
            print(f"  Bot edits: {metrics['bot_edits']}")
            if metrics['last_edit_time']:
                print(f"  Last edit: {metrics['last_edit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "="*70 + "\n")
    
    def start_monitoring(self, duration_seconds=None):
        """
        Start monitoring Wikipedia event stream
        
        Args:
            duration_seconds (int): How long to monitor (None for infinite)
        """
        print(f"\n{'='*70}")
        print("Starting Wikipedia Event Stream Monitoring")
        print(f"{'='*70}")
        print(f"Monitoring entities: {', '.join(self.entities)}")
        print(f"Database: {self.db_path}")
        print(f"Alerts: {self.alert_path}")
        if duration_seconds:
            print(f"Duration: {duration_seconds} seconds")
        else:
            print("Duration: Infinite (press Ctrl+C to stop)")
        print(f"{'='*70}\n")
        
        # Wikimedia EventStreams URL
        url = 'https://stream.wikimedia.org/v2/stream/recentchange'
        
        start_time = time.time()
        
        try:
            # Connect to event stream
            print("Connecting to Wikimedia EventStreams...")
            response = requests.get(url, stream=True, timeout=10)
            client = SSEClient(response)
            
            print("✓ Connected! Monitoring events...\n")
            
            # Process events
            for event in client:
                if event.data:
                    try:
                        event_data = json.loads(event.data)
                        
                        # Only process edit events
                        if event_data.get('type') == 'edit':
                            self.process_event(event_data)
                    
                    except json.JSONDecodeError:
                        continue
                
                # Check if duration exceeded
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    print(f"\nMonitoring duration ({duration_seconds}s) reached. Stopping...")
                    break
                
                # Periodically save metrics (every 60 seconds)
                if int(time.time() - start_time) % 60 == 0:
                    self.save_metrics()
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user (Ctrl+C)")
        
        except Exception as e:
            print(f"\nError during monitoring: {e}")
        
        finally:
            # Save final metrics
            print("\nSaving final metrics...")
            self.save_metrics()
            self.print_metrics_summary()
            print(f"\n✓ Metrics saved to: {self.db_path}")
            print(f"✓ Alerts saved to: {self.alert_path}")
            print(f"✓ Total alerts generated: {len(self.alerts)}")


def main():
    """Main function to run the stream processor"""
    
    # Define entities to monitor
    # These should be selected from IMDB analysis
    # Example: top directors, specific movies, genres
    entities = [
        "Christopher Nolan",
        "The Shawshank Redemption",
        "Quentin Tarantino",
        "The Godfather",
        "Steven Spielberg"
    ]
    
    # Create processor
    processor = WikipediaStreamProcessor(entities)
    
    # Start monitoring
    # Set duration_seconds=300 for 5 minutes test, or None for infinite
    processor.start_monitoring(duration_seconds=None)


if __name__ == "__main__":
    main()
