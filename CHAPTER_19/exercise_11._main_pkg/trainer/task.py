#! /usr/bin/env python3

from google.cloud import aiplatform
import os 

PROJECT_ID = '869083780811'
LOCATION_ID = 'us-central1'
BUCKET_URI = 'gs://job_factory_2'

aiplatform.init(project=PROJECT_ID,location=LOCATION_ID, staging_bucket=BUCKET_URI)

job = aiplatform.CustomJob.from_local_script(
    display_name='first_custom_job',
    script_path="local_script.py",
    container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest',
    replica_count=1, 
)

job.run()
