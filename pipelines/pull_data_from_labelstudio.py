"""
Use this to pull data from label studio
raw data is annotated videos
we need to convert to images and save to S3
"""

from label_studio_sdk import Client

API_ROOT = "http://18.218.125.193:8080"

id = "1"
exportType = "JSON"
api_token = "4498d6dbce82095f126fc6b6b594f21342e1351c"
output_filename = 'may14annotations.json'

# Initialize the Label Studio SDK client with the additional headers
ls = Client(url=API_ROOT, api_key=api_token)

# Get the project
project = ls.get_project(id)

# Export tasks
tasks = project.export_tasks(export_type='JSON', download_all_tasks=True)

# csv
csv_path = r'C:\Users\Administrator\Desktop\notebooks-for-ml-ops\may14annos.csv'
project.export_tasks(export_type='CSV', export_location=csv_path)