
from dotenv import load_dotenv
from upstash_redis import Redis
from uuid import uuid4

load_dotenv()
redis = Redis.from_env()

key = f"agent:test:list:{uuid4()}"

try:
    print("1. Test key:", key)

    # Push a test value.
    push_result = redis.rpush(key, "hello")
    print("2. RPUSH result:", repr(push_result))

    # Read the value back.
    list_result = redis.lrange(key, 0, -1)
    print("3. LRANGE result:", repr(list_result))

    # Check if the key exists.
    exists_result = redis.exists(key)
    print("4. EXISTS result:", repr(exists_result))

    # Clean up the test key.
    delete_result = redis.delete(key)
    print("5. DELETE result:", repr(delete_result))

except Exception as error:
    print("Redis test failed:", repr(error))
    raise
