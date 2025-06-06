_config = {
'use_fp8': False,
'cpu_offload': False
}

def get_config():

    return _config.get("use_fp8", False), _config.get("cpu_offload", False)

def update_config(new_settings):

    _config.update(new_settings)
