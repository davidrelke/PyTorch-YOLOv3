# https://drive.google.com/file/d/1OrYrD0VzLjq9xjvtOF371QZ1F4FTuwH5/view?usp=sharing
fileId=1OrYrD0VzLjq9xjvtOF371QZ1F4FTuwH5
fileName=darknet53.conv.74
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}