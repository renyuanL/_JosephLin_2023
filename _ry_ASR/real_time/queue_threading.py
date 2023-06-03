import threading
import queue
import time

q= queue.Queue(maxsize= 1024)

def worker():
    while True:
        item= q.get()
        print(f'q.get.. {item}')
        
        q.task_done()
        print(f'q.task_done.. {item}')

# Turn-on the worker thread.
threading.Thread(
    target= worker, 
    daemon= True
    ).start()

# Send thirty task requests to the worker.
for item in range(100):
    q.put(item)

# Block until all tasks are done.
q.join()

print('All work completed')
