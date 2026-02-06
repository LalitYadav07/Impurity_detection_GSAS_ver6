import GSASII
print(f"GSASII_FILE: {GSASII.__file__}")
try:
    import GSASII.pyspg
    print("pyspg: OK")
except Exception as e:
    print(f"pyspg: FAILED ({e})")
