FROM pytorch/torchserve:0.4.0-cpu

COPY handler.py conditional_motion_generator /home/model-server/

RUN pip install --no-cache-dir transformers 

RUN torch-model-archiver \
    --model-name=motiongenerator \
    --version=1.0 \
    --serialized-file=/home/model-server/pytorch_model.bin \
    --handler=/home/model-server/handler.py \
    --extra-files "/home/model-server/config.json" \
    --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--model-store", \
     "/home/model-server/model-store", \
     "--models", \
     "motiongenerator=motiongenerator.mar"]
