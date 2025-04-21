#! /usr/bin/env python3 

"""
    Q. 
        Train a model and deploy it to TF-Serving or Google Cloud AI Platform.
        Write the client code to query it using the REST API or the gRPC API.
        Update the model and deploy the new version. Your client code will now query the new version. Roll back to the first version. 
    
    A. 
        Google Cloud Vertex AI Platform used to query model
"""

from google.cloud import aiplatform
import os 

PROJECT_ID = '869083780811'
ENDPOINT_ID = '1941215266629222400'  # deployed on version 1
ENDPOINT_ID2 = '8611890739695058944' # deployed on version 6
LOCATION_ID = 'us-central1'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'cred.json')

def endpoint_predict_sample( project: str, location: str, instances: list, endpoint: str):
    aiplatform.init(project=project, location=location, staging_bucket='gs://my_staging_bucket')
    endpoint = aiplatform.Endpoint(endpoint)
    prediction = endpoint.predict(instances=instances)
    return prediction

instances = [
    
    [ 0.30, 0.20, 0.20, 0.10, 0.20, 0.10, 0.20, 0.110 ], 
    
    [ -1.371715  ,  0.58048093,  0.14552322,  0.30510637, 1.4215977 ,0.57935536, -0.30329934,  1.4869808 ],
    
    [ 1.8956385 , -0.45161435,  0.46339658, -0.46701705,  0.24996527,-0.8164176 ,  1.0037286 ,  1.355299  ]
    
]

if __name__ == "__main__":
    ret = endpoint_predict_sample(project=PROJECT_ID, location= LOCATION_ID, instances=instances, endpoint=ENDPOINT_ID)
    print(ret) 

"""

Google Cloud Model

Model ID:   
    814655652809932800

Name: 
    lunar_lander_model 

Version:
    1 

ENDPOINT_ID:
    1941215266629222400

  
SDK Output:

    Prediction(predictions=[[0.312848359, 0.236094132, 0.217715696, 0.233341739], [0.414523602, 0.112857334, 0.306751847, 0.16586718], [0.319322735, 0.398763537, 0.159842864, 0.122070901]], deployed_model_id='8489429333616623616', metadata=None, model_version_id='1', model_resource_name='projects/869083780811/locations/us-central1/models/814655652809932800', explanations=None)

"""

"""
Google Cloud Model

Model ID:
    814655652809932800

Name: 
    lunar_lander_model 

Version:
  6

ENDPOINT_ID:
    8611890739695058944
  
SDK Output:

    Prediction(predictions=[[0.262676597, 0.271419078, 0.255823731, 0.210080639], [0.269151241, 0.29415518, 0.262208253, 0.174485326], [0.26882872, 0.209912121, 0.406539112, 0.114720076]], deployed_model_id='214346493300047872', metadata=None, model_version_id='6', model_resource_name='projects/869083780811/locations/us-central1/models/814655652809932800', explanations=None)


"""
