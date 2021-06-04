FROM pytorch/torchserve:0.4.0-cpu

COPY handler.py requirements.txt conditional_motion_generator /home/model-server/

RUN pip install -r /home/model-server/requirements.txt

RUN torch-model-archiver \
  --model-name=motiongenerator \
  --version=1.0 \
  --serialized-file=/home/model-server/conditional_motion_generator/pytorch_model.bin \
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--models", \
     "motiongenerator=motiongenerator.mar"]
