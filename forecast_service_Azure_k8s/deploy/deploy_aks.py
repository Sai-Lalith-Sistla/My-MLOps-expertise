from azureml.core import Workspace, Model, Environment, InferenceConfig
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice

# Load workspace
ws = Workspace.from_config('../config/config.json')

# Define AKS cluster (or attach existing one)
aks_name = 'aks-forecast-cluster'
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print("Found existing AKS cluster.")
except:
    print("Creating new AKS cluster...")
    prov_config = AksCompute.provisioning_configuration(vm_size="Standard_DS3_v2")
    aks_target = ComputeTarget.create(workspace=ws, name=aks_name, provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)

# Define environment
env = Environment.from_conda_specification(name='forecast-env', file_path='../deploy/environment.yml')

# Define inference configuration
inference_config = InferenceConfig(
    environment=env,
    source_directory='../deploy',
    entry_script='score.py'
)

# Fetch model from registry
model = Model(ws, name='forecast-model')

# AKS deployment configuration
aks_config = AksWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4,
    enable_app_insights=True,  # Enables monitoring
    autoscale_enabled=True,
    autoscale_min_replicas=1,
    autoscale_max_replicas=3,
    collect_model_data=True
)

# Deploy the model
service = Model.deploy(
    workspace=ws,
    name='forecast-service-aks',
    models=[model],
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target=aks_target,
    overwrite=True
)

service.wait_for_deployment(show_output=True)

print(f"Service State: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
