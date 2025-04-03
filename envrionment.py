import os
import gymnasium as gym
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    return 2 * intersection / union if union > 0 else 0.0

def normalize_mri(image):
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / (std + 1e-8)




def resample_image(image, target_shape):
    factors = [t / s for s, t in zip(image.shape, target_shape)]
    return zoom(image, factors, order=1)




class CryoAblationEnv(gym.Env):
    def __init__(self, patient_dirs, needle_diameter_mm=3, target_shape=(128, 128, 128),
                 max_insertions=4):
        
        super(CryoAblationEnv, self).__init__()
        self.patient_dirs = patient_dirs
        self.needle_diameter_mm = needle_diameter_mm
        self.target_shape = target_shape
        self.max_insertions = max_insertions
        self.reward_scale = reward_scale  

       



        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,3), dtype=np.float32)
        obs_shape = (*self.target_shape, 4)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self.patient_data = []
        for patient_dir in self.patient_dirs:
            mri_image, cancer_mask, gland_mask = self.load_patient_data(patient_dir)
            self.patient_data.append((mri_image, cancer_mask, gland_mask))




        self.needle_radius = None
        if self.patient_dirs:
            t2_path = os.path.join(self.patient_dirs[0], "t2.nii.gz")
            voxel_spacing = nib.load(t2_path).header.get_zooms()[:3]
            radius_mm = self.needle_diameter_mm / 2.0
            radius_voxels = np.array([radius_mm / vs for vs in voxel_spacing])
            self.needle_radius = np.mean(radius_voxels)

        self.needle_mask = None
        self.insertions = 0

    def load_patient_data(self, patient_dir):
        t2_path = os.path.join(patient_dir, "t2.nii.gz")
        adc_path = os.path.join(patient_dir, "adc.nii.gz")
        dwi_path = os.path.join(patient_dir, "dwi.nii.gz")
        cancer_path = os.path.join(patient_dir, "l_a1.nii.gz")  
        gland_path = os.path.join(patient_dir, "gland.nii.gz")

     


        t2_img = nib.load(t2_path).get_fdata()
        adc_img = nib.load(adc_path).get_fdata()
        dwi_img = nib.load(dwi_path).get_fdata()

        t2_resampled = normalize_mri(resample_image(t2_img, self.target_shape))
        adc_resampled = normalize_mri(resample_image(adc_img, self.target_shape))
        dwi_resampled = normalize_mri(resample_image(dwi_img, self.target_shape))

        mri_image = np.stack([t2_resampled, adc_resampled, dwi_resampled], axis=-1)

       


        cancer_data = nib.load(cancer_path).get_fdata()
        cancer_mask = (resample_image(cancer_data, self.target_shape) > 0).astype(np.float32)

        gland_data = nib.load(gland_path).get_fdata()
        gland_mask = (resample_image(gland_data, self.target_shape) > 0).astype(np.float32)

        return mri_image, cancer_mask, gland_mask

    def create_spherical_mask(self, center):
       
        grid = np.ogrid[:self.target_shape[0], :self.target_shape[1], :self.target_shape[2]]
        dist = np.sqrt((grid[0] - center[0])**2 +
                       (grid[1] - center[1])**2 +
                       (grid[2] - center[2])**2)
        mask = (dist <= self.needle_radius).astype(np.float32)  
        
        
        
        return mask

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.patient_data))
        self.mri_image, self.cancer_mask, self.gland_mask = self.patient_data[idx]
        self.needle_mask = np.zeros(self.target_shape, dtype=np.float32)
        self.insertions = 0

        obs = np.concatenate([self.mri_image, np.expand_dims(self.needle_mask, axis=-1)], axis=-1)
        return obs, {}




    def step(self, action):

        voxel_coord = (action * (np.array(self.target_shape) - 1)).astype(int)
        new_mask = self.create_spherical_mask(voxel_coord)

        non_overlap_mask = new_mask * (1 - self.needle_mask)
        self.needle_mask = np.maximum(self.needle_mask, non_overlap_mask)
        cancer_dice = dice_score(self.cancer_mask, self.needle_mask)





        cancer_indices = np.array(np.where(self.cancer_mask))
        gland_indices = np.array(np.where(self.gland_mask))
        cancer_center = cancer_indices.mean(axis=1) if cancer_indices.size > 0 else np.zeros(3)
        gland_center = gland_indices.mean(axis=1) if gland_indices.size > 0 else np.zeros(3)


        cancer_distance = np.linalg.norm(voxel_coord - cancer_center)
        gland_distance = np.linalg.norm(voxel_coord - gland_center)
        proximity_penalty = 0.01 * (cancer_distance + gland_distance)



        reward = self.reward_scale * (cancer_dice - proximity_penalty)

        self.insertions += 1

       


        obs = np.concatenate([self.mri_image, np.expand_dims(self.needle_mask, axis=-1)], axis=-1)
        terminated = self.insertions >= self.max_insertions
        truncated = False

        info = {
            "cancer_dice": cancer_dice,
            "insertions": self.insertions,
            "cancer_distance": cancer_distance,
            "gland_distance": gland_distance
        }
        return obs, reward, terminated, truncated, info
