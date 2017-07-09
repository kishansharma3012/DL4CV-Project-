cd ../../datasets/train
mkdir -p train val test temp

for i in {0..37}; do mkdir train/c_$i; done
for i in {0..37}; do mkdir val/c_$i; done
for i in {0..37}; do mkdir test/c_$i; done

mv c_* temp
cd temp

find . -iname *.jpg | shuf | head -n 600| xargs -I{} mv {} ../train/{}
find . -iname *.jpg | shuf | head -n 100| xargs -I{} mv {} ../val/{}
find . -iname *.jpg | shuf | head -n 100| xargs -I{} mv {} ../test/{}
