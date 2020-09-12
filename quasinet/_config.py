"""Global configuration state and functions for management

TODO: allow for config values to be set by users
"""

_global_config = {
    'nan_value': '',
}


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`

    Parameters
    ----------
    None

    Returns
    -------
    config : dict
        Dictionary mapping config key to its value
    """

    return _global_config.copy()