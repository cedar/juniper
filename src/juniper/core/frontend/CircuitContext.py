current = None

def get_current():
    return current

def set_current(circuit):
    global current
    current = circuit
