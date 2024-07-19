import sys
import os

# Ensure the project root is added to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, project_root)
print("Updated sys.path:", sys.path)

try:
    from components.data_transformation import DataTransformation
    from components.data_transformation import DataTransformationConfig
    print("Import successful!")
except ImportError as e:
    print(f"ImportError: {e}")