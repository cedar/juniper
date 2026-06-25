import logging

logger = logging.getLogger(__name__)
current = None

def get_current():
    return current

def set_current(circuit):
    global current
    current = circuit
