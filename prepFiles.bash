cd /home/drew/done/megaTest;

# extract all
find -iname "*.epub" -exec 7z x {} -aoa > /dev/null 2>&1\;

# cleanup other files than .xhtml
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