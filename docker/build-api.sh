# cd into proj. root
cd $(dirname $0)
cd ..
docker build -t lukasblecher/pix2tex:api -f docker/api.dockerfile .
