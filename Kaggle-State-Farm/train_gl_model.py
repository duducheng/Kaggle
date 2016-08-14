import graphlab as gl
from utils import train_set
from gl_utils import get_sf
from config import GL_MODEL

gl_checkpoints = '.gl_checkpoints/'

train_sframe = get_sf(train_set, times=1)

model = gl.boosted_trees_classifier.create(dataset=train_sframe, target='classname', features=['conv'], max_iterations=600,
                                           max_depth=3, step_size=0.1, metric=['accuracy', 'log_loss'],
                                           early_stopping_rounds=200, model_checkpoint_path=gl_checkpoints + GL_MODEL,
                                           model_checkpoint_interval=50)
model.save(GL_MODEL)
