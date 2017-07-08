cd Desktop/datasets/train
mkdir -p train val test
for i in {0..37}; do mkdir val/c_$i; done
for i in {0..37}; do mkdir test/c_$i; done
mv c_* train
cd train
find . -iname *.jpg | shuf | head -n 2100| xargs -I{} mv {} ../val/{}
find . -iname *.jpg | shuf | head -n 2100| xargs -I{} mv {} ../test/{}
