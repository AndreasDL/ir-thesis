cd /home/drew/done/megaTest;

# extract all
find -iname "*.epub" -exec 7z x {} -aoa \;

# cleanup
mkdir /tmp/files;
mv "*.xhtml" /tmp/files \;
rm -rvf *;
mv /tmp/files/* .;
rmdir /tmp/files;

#cleanup in files
#html tags
find -iname '*.xhtml' -exec sed -i -e 's/<[^>]*>//g' {} \;
#other chars
find -iname '*.xhtml' -exec sed -i -e 's/[^a-zA-Z0-9 ]//g' {} \;

cd -;
./a.out freqs.txt;