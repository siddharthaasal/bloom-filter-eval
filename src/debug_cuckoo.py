from filters import CuckooFilterWrapper
import config

try:
    f = CuckooFilterWrapper(capacity=1000)
    print(f"Capacity: {f.capacity}")
    print(f"Filter object: {f.filter}")
    print(f"Has capacity attr: {hasattr(f.filter, 'capacity')}")
    if hasattr(f.filter, "capacity"):
        val = getattr(f.filter, "capacity")
        print(f"capacity value: {val}, type: {type(val)}")

    print(f"Size Bits: {f.size_bits()}")
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"Error: {e}")
