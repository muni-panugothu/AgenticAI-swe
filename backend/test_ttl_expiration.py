
import time
from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv()


redis = Redis.from_env()


# Use a unique key so this test doesn't affect existing memories.

key = "agent:test:tt1:expiration"

print("Step 1: Saving a test value with a 10-second TTL...")

redis.set(key, "This memory should expires automatically", ex=10)

print("Step 2: Checking immediately...")
value_now = redis.get(key)
print("Value immediately after saving:", value_now)

print("Step 3: Waiting 12 seconds...")
time.sleep(12)

print("Step 4: Checking after expiry...")
value_after = redis.get(key)
print("Value after expiry:", value_after)


if value_after is None:
    print("SUCCESS: Redis automatically expired the test value.")
else:
    print("FAILED: The test value still exists.")


    