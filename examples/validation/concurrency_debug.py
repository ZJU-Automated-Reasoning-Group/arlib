"""
Predictive Trace Analysis (A Demo)

Analyze the trace of Java programs to predict whether there exist potential scheduling where the program may have data race bugs.

Refer to the papers of Jeff Huang, Jens Palsberg, etc.

TBD: implement an SMT-based approach
"""

import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass
class Event:
    thread_id: int
    operation: str  # 'read', 'write', 'lock', 'unlock'
    variable: str
    timestamp: int

class PredictiveRaceDetector:
    def __init__(self):
        self.events = []
        self.happens_before_graph = nx.DiGraph()
        self.lock_to_events = {}
        self.var_to_events = {}
        self.threads = set()
        
    def add_event(self, event: Event):
        """Add an event to the trace"""
        self.events.append(event)
        self.threads.add(event.thread_id)
        
        # Update variable and lock maps
        if event.operation in ['read', 'write']:
            if event.variable not in self.var_to_events:
                self.var_to_events[event.variable] = []
            self.var_to_events[event.variable].append(event)
        elif event.operation in ['lock', 'unlock']:
            if event.variable not in self.lock_to_events:
                self.lock_to_events[event.variable] = []
            self.lock_to_events[event.variable].append(event)
    
    def build_happens_before(self):
        """
        Build the happens-before relation graph.
        
        The happens-before relation is a partial order of events that captures
        the causal relationships between them. Two types of edges are added:
        1. Program order edges: Events from the same thread are ordered by their timestamps
        2. Synchronization edges: Unlock operations happen-before subsequent lock operations 
           on the same lock object
        """
        # Initialize graph with program order edges (events within the same thread)
        for thread in self.threads:
            thread_events = [e for e in self.events if e.thread_id == thread]
            thread_events.sort(key=lambda e: e.timestamp)
            
            for i in range(len(thread_events) - 1):
                self.happens_before_graph.add_edge(id(thread_events[i]), id(thread_events[i+1]))
        
        # Add lock-based happens-before edges (synchronization between threads)
        for lock, lock_events in self.lock_to_events.items():
            lock_events.sort(key=lambda e: e.timestamp)
            
            for i in range(len(lock_events) - 1):
                if lock_events[i].operation == 'unlock' and lock_events[i+1].operation == 'lock':
                    self.happens_before_graph.add_edge(id(lock_events[i]), id(lock_events[i+1]))
    
    def detect_races(self) -> List[Tuple[Event, Event]]:
        """
        Detect potential data races.
        
        A data race occurs when two memory accesses:
        1. Access the same memory location
        2. At least one is a write
        3. Are from different threads
        4. Are not ordered by happens-before relation (can execute in any order)
        """
        self.build_happens_before()
        races = []
        
        # Check for races between memory accesses
        for var, var_events in self.var_to_events.items():
            write_events = [e for e in var_events if e.operation == 'write']
            all_events = var_events
            
            for i, e1 in enumerate(all_events):
                for e2 in all_events[i+1:]:
                    # Skip if both are reads (no race if both threads are just reading)
                    if e1.operation == 'read' and e2.operation == 'read':
                        continue
                    
                    # Skip if same thread (no race within the same thread)
                    if e1.thread_id == e2.thread_id:
                        continue
                    
                    # Check if not ordered by happens-before
                    # If neither e1→e2 nor e2→e1, then they are concurrent and can race
                    if not nx.has_path(self.happens_before_graph, id(e1), id(e2)) and \
                       not nx.has_path(self.happens_before_graph, id(e2), id(e1)):
                        races.append((e1, e2))
        
        return races


def demo():
    """
    Demonstrate predictive race detection with a simple example.
    
    This example has two threads accessing shared variables x and y.
    Some accesses are protected by locks, but there's a potential
    race condition between Thread 1's write to y and Thread 2's read of y.
    """
    # Create a sample trace
    detector = PredictiveRaceDetector()
    
    # Thread 1 events
    detector.add_event(Event(1, 'lock', 'lock1', 1))    # Thread 1 acquires lock1
    detector.add_event(Event(1, 'read', 'x', 2))        # Thread 1 reads x (protected)
    detector.add_event(Event(1, 'unlock', 'lock1', 3))  # Thread 1 releases lock1
    detector.add_event(Event(1, 'lock', 'lock2', 4))    # Thread 1 acquires lock2
    detector.add_event(Event(1, 'write', 'y', 5))       # Thread 1 writes to y (protected by lock2)
    detector.add_event(Event(1, 'unlock', 'lock2', 6))  # Thread 1 releases lock2
    
    # Thread 2 events
    detector.add_event(Event(2, 'lock', 'lock1', 7))    # Thread 2 acquires lock1
    detector.add_event(Event(2, 'write', 'x', 8))       # Thread 2 writes to x (protected)
    detector.add_event(Event(2, 'unlock', 'lock1', 9))  # Thread 2 releases lock1
    detector.add_event(Event(2, 'read', 'y', 10))       # Thread 2 reads y (unprotected!)
    
    # Detect races
    races = detector.detect_races()
    
    print(f"Found {len(races)} potential data races:")
    for i, (e1, e2) in enumerate(races):
        print(f"Race {i+1}:")
        print(f"  Event 1: Thread {e1.thread_id}, {e1.operation} on {e1.variable} at time {e1.timestamp}")
        print(f"  Event 2: Thread {e2.thread_id}, {e2.operation} on {e2.variable} at time {e2.timestamp}")
    
    print("\nExplanation:")
    print("This demo shows predictive race detection for concurrent programs.")
    print("It uses a happens-before graph from the execution trace.")
    print("Races happen when memory accesses from different threads")
    print("aren't ordered by happens-before, meaning they could execute")
    print("in any order, causing bugs.")
    print("\nIn this example, Thread 1 writes to 'y' and Thread 2 reads 'y'")
    print("without proper lock protection. Thread 2 should use lock2.")


if __name__ == "__main__":
    demo() 
    