# https://drive.google.com/file/d/1saXnJD2HuGFw3JZGgVqb5RaldH_mg5fN/view?usp=sharing
fileId=1saXnJD2HuGFw3JZGgVqb5RaldH_mg5fN
fileName=yolov3-carla-29.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}