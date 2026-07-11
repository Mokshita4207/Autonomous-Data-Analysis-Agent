from e2b_code_interpreter import Sandbox
from config.e2b_config import E2B_API_KEY

with Sandbox.create(api_key=E2B_API_KEY) as sandbox:

    execution = sandbox.run_code("""
print("Hello")
""")

    print(type(execution))
    print(execution)
    print(dir(execution))