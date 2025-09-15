from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams, VolumeMount, Sandbox
try:
    from daytona import Resources, CreateSandboxFromImageParams, Image  
except Exception: 
    Resources = None  
    CreateSandboxFromImageParams = None  # type: ignore
    Image = None  # type: ignore
from config import DAYTONA_KEY
from typing import Dict, Any

daytona = Daytona(DaytonaConfig(api_key=DAYTONA_KEY))

persistent_sandboxes: Dict[str, Any] = {}
persistent_volumes: Dict[str, Any] = {}
# Ephemeral sandboxes (e.g., per sub-trainer) keyed by (job_id, key)
ephemeral_sandboxes: Dict[tuple[str, str], Any] = {}


def create_volume(job_id: str):
    try:
        volume = daytona.volume.get(f"sproutml-job-{job_id}", create=True)
        persistent_volumes[job_id] = volume
        return volume
    except Exception as e:
        error_msg = str(e)
        print(f"Error creating volume for job {job_id}: {error_msg}")
        
        # Check if it's a quota exceeded error
        if "disk quota exceeded" in error_msg.lower() or "maximum allowed" in error_msg.lower():
            print(f"Disk quota exceeded for job {job_id}. Attempting cleanup of old volumes...")
            cleanup_old_volumes()
            
            # Try again after cleanup
            try:
                volume = daytona.volume.get(f"sproutml-job-{job_id}", create=True)
                persistent_volumes[job_id] = volume
                return volume
            except Exception as retry_e:
                print(f"Volume creation failed even after cleanup for job {job_id}: {retry_e}")
                return None
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
    if volume is None:
        print(f"No volume available for job {job_id}, creating sandbox without persistent volume")
        # Create sandbox without volume mount as fallback
        params = CreateSandboxFromSnapshotParams(language="python")
    else:
        params = CreateSandboxFromSnapshotParams(
            language="python",
            volumes=[VolumeMount(volumeId=volume.id, mountPath="/home/daytona/volume")],
        )
    
    try:
        sandbox = daytona.create(params)
        persistent_sandboxes[job_id] = sandbox
        return sandbox
    except Exception as e:
        error_msg = str(e)
        print(f"Error creating sandbox for job {job_id}: {error_msg}")
        
        # Check if it's a quota exceeded error
        if "disk quota exceeded" in error_msg.lower() or "maximum allowed" in error_msg.lower():
            print(f"Disk quota exceeded for job {job_id}. Attempting cleanup and fallback...")
            cleanup_old_sandboxes()
            
            # Try creating without volume as fallback
            try:
                fallback_params = CreateSandboxFromSnapshotParams(language="python")
                sandbox = daytona.create(fallback_params)
                persistent_sandboxes[job_id] = sandbox
                print(f"Created fallback sandbox (no persistent volume) for job {job_id}")
                return sandbox
            except Exception as fallback_e:
                print(f"Fallback sandbox creation failed for job {job_id}: {fallback_e}")
                return None
        return None

def create_ephemeral_sandbox(job_id: str, key: str, cpu: int = 2, memory: int = 4, disk: int = 3):
    """Create or return an ephemeral sandbox for a sub-task under a job.
    Tries to allocate more resources if supported by the SDK, else falls back to snapshot params.
    """
    # Reuse if already exists
    if (job_id, key) in ephemeral_sandboxes:
        return ephemeral_sandboxes[(job_id, key)]

    volume = get_volume(job_id)
    try:
        if Resources and CreateSandboxFromImageParams and Image:
            resources = Resources(cpu=cpu, memory=memory, disk=disk)  # type: ignore
            params = CreateSandboxFromImageParams(  # type: ignore
                image=Image.debian_slim("3.12"),  # lightweight image
                resources=resources,
            )
            sandbox = daytona.create(params)
            # Mount volume if available
            if volume is not None and hasattr(sandbox, "mount_volume"):
                try:
                    sandbox.mount_volume(volume.id, "/home/daytona/volume")  # type: ignore
                except Exception:
                    pass
        else:
            # Fallback to snapshot params (no custom resources)
            if volume is None:
                params = CreateSandboxFromSnapshotParams(language="python")
            else:
                params = CreateSandboxFromSnapshotParams(
                    language="python",
                    volumes=[VolumeMount(volumeId=volume.id, mountPath="/home/daytona/volume")],
                )
            sandbox = daytona.create(params)
        ephemeral_sandboxes[(job_id, key)] = sandbox
        return sandbox
    except Exception as e:
        print(f"Error creating ephemeral sandbox for {job_id}:{key}: {e}")
        return None

def get_ephemeral_sandbox(job_id: str, key: str):
    try:
        return ephemeral_sandboxes[(job_id, key)]
    except Exception:
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

def cleanup_old_volumes():
    """Clean up old volumes to free disk quota"""
    try:
        # Get list of all volumes
        volumes = daytona.volume.list()
        sproutml_volumes = [v for v in volumes if v.name.startswith("sproutml-job-")]
        
        # Sort by creation time and keep only the 5 most recent
        sorted_volumes = sorted(sproutml_volumes, key=lambda v: getattr(v, 'created_at', ''), reverse=True)
        volumes_to_delete = sorted_volumes[5:]  # Delete all but the 5 most recent
        
        for volume in volumes_to_delete:
            try:
                print(f"Cleaning up old volume: {volume.name}")
                daytona.volume.delete(volume.id)
            except Exception as e:
                print(f"Failed to delete volume {volume.name}: {e}")
                
        print(f"Cleaned up {len(volumes_to_delete)} old volumes")
        
    except Exception as e:
        print(f"Error during volume cleanup: {e}")

def cleanup_old_sandboxes():
    """Clean up old sandboxes to free resources"""
    try:
        # Get list of all sandboxes
        sandboxes = daytona.sandbox.list()
        sproutml_sandboxes = [s for s in sandboxes if hasattr(s, 'name') and 'sproutml' in s.name.lower()]
        
        # Sort by creation time and keep only the 3 most recent
        sorted_sandboxes = sorted(sproutml_sandboxes, key=lambda s: getattr(s, 'created_at', ''), reverse=True)
        sandboxes_to_delete = sorted_sandboxes[3:]  # Delete all but the 3 most recent
        
        for sandbox in sandboxes_to_delete:
            try:
                print(f"Cleaning up old sandbox: {getattr(sandbox, 'name', sandbox.id)}")
                daytona.sandbox.delete(sandbox.id)
            except Exception as e:
                print(f"Failed to delete sandbox {getattr(sandbox, 'name', sandbox.id)}: {e}")
                
        print(f"Cleaned up {len(sandboxes_to_delete)} old sandboxes")
        
    except Exception as e:
        print(f"Error during sandbox cleanup: {e}")

def force_cleanup_job_resources(job_id: str):
    """Force cleanup of all resources for a specific job"""
    try:
        # Clean up sandbox
        if job_id in persistent_sandboxes:
            try:
                sandbox = persistent_sandboxes[job_id]
                if sandbox:
                    sandbox.delete()
                del persistent_sandboxes[job_id]
                print(f"Cleaned up sandbox for job {job_id}")
            except Exception as e:
                print(f"Error cleaning up sandbox for job {job_id}: {e}")
        
        # Clean up volume
        if job_id in persistent_volumes:
            try:
                delete_volume(job_id)
                del persistent_volumes[job_id]
                print(f"Cleaned up volume for job {job_id}")
            except Exception as e:
                print(f"Error cleaning up volume for job {job_id}: {e}")
                
    except Exception as e:
        print(f"Error during force cleanup for job {job_id}: {e}")