

import qnet


def load_qnet(qnet_type, extra_descriptions):
    """Load the pre-trained qnet.
    """
    
    qnet_type = qnet_type.lower()
    all_qnet_types = ['coronavirus', 'influenza']
    
    if qnet_type not in all_qnet_types:
        raise ValueError("`qnet_type`: {} is not in {}".format(qnet_type, all_qnet_types)
                         
    if qnet_type == 'coronavirus':
                         
                         
    