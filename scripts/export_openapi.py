"""
Export OpenAPI Schema
Utility to generate openapi.json for Custom GPT Actions.
"""
import json
from api_app import app

def export_schema():
    schema = app.openapi()
    
    # Optional: Customize schema for GPT
    # e.g., removal of complex types or addition of clear descriptions
    
    with open("openapi_v7.1.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("Exported openapi_v7.1.json")

if __name__ == "__main__":
    export_schema()
