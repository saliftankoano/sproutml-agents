from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams, VolumeMount, Sandbox
from config import DAYTONA_KEY
from typing import Dict, Any

daytona = Daytona(DaytonaConfig(api_key=DAYTONA_KEY))

persistent_sandboxes: Dict[str, Any] = {}
persistent_volumes: Dict[str, Any] = {}


def create_volume(job_id: str):
    try:
        volume = daytona.volume.get(f"sproutml-job-{job_id}", create=True)
        persistent_volumes[job_id] = volume
        return volume
    except Exception as e:
        print(f"Error creating volume for job {job_id}: {e}")
        return None

def wait_for_volume_ready(job_id: str, max_wait_time: int = 60):
    """Wait for volume to be in ready state"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            volume = daytona.volume.get(f"sproutml-job-{job_id}")
            if hasattr(volume, 'state') and volume.state == 'ready':
                return volume
            elif hasattr(volume, 'status') and volume.status == 'ready':
                return volume
            print(f"Volume {job_id} not ready yet, waiting...")
            time.sleep(2)
        except Exception as e:
            print(f"Error checking volume status for job {job_id}: {e}")
            time.sleep(2)
    
    print(f"Volume {job_id} did not become ready within {max_wait_time} seconds")
    return None

def get_volume(job_id: str):
    try:
        volume = daytona.volume.get(f"sproutml-job-{job_id}")
        persistent_volumes[job_id] = volume
        return volume
    except Exception as e:
        print(f"Error getting volume for job {job_id}: {e}")
        return None

def delete_volume(job_id: str):
    try:
        volume = daytona.volume.delete(f"sproutml-job-{job_id}")
        persistent_volumes[job_id] = None
        return volume
    except Exception as e:
        print(f"Error deleting volume for job {job_id}: {e}")
        return None

def create_sandbox(job_id: str):
    volume = get_volume(job_id)
    params = CreateSandboxFromSnapshotParams(
        language="python",
        volumes=[VolumeMount(volumeId=volume.id, mountPath="/home/daytona/volume")],
    )
    try:
        sandbox = daytona.create(params)
        persistent_sandboxes[job_id] = sandbox
        return sandbox
    except Exception as e:
        print(f"Error creating sandbox for job {job_id}: {e}")
        return None

def get_sandbox(job_id: str):
    try:
        sandbox = daytona.sandbox.get(f"sproutml-job-{job_id}")
        persistent_sandboxes[job_id] = sandbox
        return sandbox
    except Exception as e:
        print(f"Error getting sandbox for job {job_id}: {e}")
        return None

def delete_sandbox(job_id: str):
    try:
        sandbox = daytona.sandbox.delete(f"sproutml-job-{job_id}")
        persistent_sandboxes[job_id] = None
        return sandbox
    except Exception as e:
        print(f"Error deleting sandbox for job {job_id}: {e}")
        return None

def get_persistent_sandbox(job_id: str):
    try:
        return persistent_sandboxes[job_id]
    except Exception as e:
        print(f"Error getting persistent sandbox for job {job_id}: {e}")
        return None

def get_persistent_volume(job_id: str):
    try:
        return persistent_volumes[job_id]
    except Exception as e:
        print(f"Error getting persistent volume for job {job_id}: {e}")
        return None