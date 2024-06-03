mkdir -p "${DATASETS}"

# download large scale ood datasets
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
wget -P ${DATASETS} https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

# download small scale ood datasets
wget -P ${DATASETS} https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
gdown 'https://drive.google.com/uc?export=download&id=1-Brox5MA287GCFX80sDY7DhBW3pqsgL5' -O $"${DATASETS}/place_subset.zip"


# places365
PLACES365_DATASET_DIR="${DATASETS}/places365"
mkdir -p ${PLACES365_DATASET_DIR}

# unpack
tar -xf ${DATASETS}/iSUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/iNaturalist.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/SUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/Places.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/dtd-r1.0.1.tar.gz -C ${DATASETS}
unzip -u ${DATASETS}/place_subset.zip -d ${PLACES365_DATASET_DIR} -x "__MACOSX/*"


# download checkpoints
#wget -P ${MODELS} https://www.dropbox.com/s/o5r3t3f0uiqdmpm/checkpoints.zip # to change
gdown 'https://drive.google.com/uc?export=download&id=1wzr4YHt8_NkY-o9ItO4GBqfntxjz-hNv' #-O "${MODELS}"
unzip ckpt.zip
rm ckpt.zip
