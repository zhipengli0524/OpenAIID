import os
file_dir = os.path.dirname(__file__)
file_dir = os.path.abspath(file_dir)
print os.path.join(file_dir, "model_params", "params,bin")
