# https://drive.google.com/file/d/1T3OBV6CLDNG7Nm9ElLZJqQ7oLmFWnNWS/view?usp=sharing
fileId=1T3OBV6CLDNG7Nm9ElLZJqQ7oLmFWnNWS
fileName=yolov3-carla-30.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}