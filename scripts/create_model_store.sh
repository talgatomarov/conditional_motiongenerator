torch-model-archiver --model-name "motiongenerator" \
                     --version 1.0 \
                     --serialized-file ./conditional_motion_generator/pytorch_model.bin \
                     --extra-files "./conditional_motion_generator/config.json" \
                     --handler "./handler.py"

mkdir model_store && mv motiongenerator.mar model_store