"""
Utils functions for the project.
"""
import uuid

def generate_uuid() -> str:
    """
    Generate a UUID string.
    """
    return str(uuid.uuid4())
