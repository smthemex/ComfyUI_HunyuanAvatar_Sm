_config = {
'use_fp8': False

}

def get_config():

    return _config.get("use_fp8", False)

def update_config(new_settings):

    _config.update(new_settings)