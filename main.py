from track import *

def program():
    while True:
        track(-1)

thread = threading.Thread(target=program)
thread.start()

video()
