import numpy as np 
        
        
contact_estimate = np.array([151., 120. ,73. , 44.])
feet_thresh = np.array([317,112,300,51])
contact_state = 1.0 * (contact_estimate > feet_thresh)
print(contact_estimate > feet_thresh)