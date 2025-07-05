# Paths
data_path = 'path/to/your/ultrasound/images' # must contain columns: 'frame_path', 'patient_id' ,'video_name' ,'frame_id' 
checkpoint_path = 'path/to/checkpoints' # where to save model checkpoints

# Hyperparameters
batch_size = 64
lr_encoder = 1e-5
lr_default = 1e-4
num_epochs = 15