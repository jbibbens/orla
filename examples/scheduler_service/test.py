import threading
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="unused")

done = []
lock = threading.Lock()

def fire(label: str, priority: int) -> None:
    client.chat.completions.create(
        model="unused",
        messages=[{"role": "user", "content": f"say {label}"}],
        extra_headers={"X-Orla-Stage": "reply", "X-Orla-Tag-Priority": str(priority)},
    )
    with lock:
        done.append(label)

threads = [
    threading.Thread(target=fire, args=("low", 1)),
    threading.Thread(target=fire, args=("high", 9)),
    threading.Thread(target=fire, args=("medium", 5)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("finished order:", done)
